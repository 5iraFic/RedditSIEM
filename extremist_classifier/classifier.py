"""
ExtremistCommentClassifier  –  main public interface.

Pipeline per comment:
  1. Split text into sentences (NLTK sent_tokenize).
  2. For each sentence: scan entity lists → (sentence, entity) pairs.
  3. For each pair:
       a. ABSAEnsemble.analyze(sentence, entity_surface)
          → sentiment + confidence + per-model details
       b. detoxify.predict(sentence)
          → toxicity / identity_attack / threat / severe_toxicity scores
  4. Determine signal type per pair, compute sentence_extremism_score.
  5. Aggregate to doc_score + risk_label.
  6. Expose ClassificationResult with full evidence + SIEM export.

Scoring formula
---------------
sentence_extremism_score =
    (absa_weight * absa_confidence + toxicity_weight * toxicity_composite)
    * signal_direction_multiplier

Where:
  signal_direction_multiplier
    = 1.0  for EXTREMIST_SIGNAL  (friendly+positive OR enemy+negative)
    = 0.0  for COUNTER_SIGNAL    (friendly+negative OR enemy+positive)
    = 0.3  for WEAK_SIGNAL       (neutral sentiment)

  toxicity_composite = mean(toxicity, severe_toxicity, identity_attack, threat)

doc_score = clip(mean(sentence_extremism_scores), 0, 1)

Risk thresholds
---------------
  >= 0.75 → HIGH
  >= 0.45 → MEDIUM
  >= 0.15 → LOW
  <  0.15 → NONE
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# Signal types
EXTREMIST_SIGNAL = "EXTREMIST_SIGNAL"
COUNTER_SIGNAL   = "COUNTER_SIGNAL"
WEAK_SIGNAL      = "WEAK_SIGNAL"

_SIGNAL_MULTIPLIER = {
    EXTREMIST_SIGNAL: 1.0,
    COUNTER_SIGNAL:   0.0,
    WEAK_SIGNAL:      0.3,
}


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SentenceResult:
    sentence: str
    sentence_index: int
    entity_label: str          # canonical name, e.g. "Islamic State"
    entity_surface: str        # matched text, e.g. "ISIS"
    entity_type: str           # "friendly" | "enemy"
    absa_sentiment: str        # "positive" | "negative" | "neutral"
    absa_confidence: float
    absa_agreement: bool       # True if DeBERTa and pyabsa agreed
    deberta_result: dict | None
    pyabsa_result: dict | None
    toxicity_scores: dict      # full detoxify output
    toxicity_composite: float
    signal_type: str           # EXTREMIST_SIGNAL | COUNTER_SIGNAL | WEAK_SIGNAL
    sentence_extremism_score: float   # [0, 1]


@dataclass
class ClassificationResult:
    comment_id: str | None
    subreddit: str | None
    author: str | None
    comment_text: str
    sentences: list[SentenceResult] = field(default_factory=list)
    doc_score: float = 0.0
    risk_label: str = "NONE"           # HIGH | MEDIUM | LOW | NONE
    entity_labels_found: list[str] = field(default_factory=list)
    friendly_entity_count: int = 0
    enemy_entity_count: int = 0
    extremist_sentence_count: int = 0
    counter_sentence_count: int = 0
    flagged_sentences: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Risk     : {self.risk_label}  (score={self.doc_score:.3f})",
            f"Entities : {self.entity_labels_found}",
            f"Flagged  : {self.extremist_sentence_count} sentence(s)",
        ]
        for s in self.sentences:
            if s.signal_type == EXTREMIST_SIGNAL:
                lines.append(
                    f"  [{s.entity_type.upper()}] {s.entity_label!r}  "
                    f"sent={s.absa_sentiment}  conf={s.absa_confidence:.2f}  "
                    f"tox={s.toxicity_composite:.2f}  "
                    f"score={s.sentence_extremism_score:.3f}"
                )
                lines.append(f"    {s.sentence[:110]!r}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def to_siem_event(self) -> dict:
        """
        Flat dict suitable for shipping to Elasticsearch / Splunk / SIEM.
        ISO-8601 UTC timestamp is injected automatically.
        """
        return {
            "timestamp":               datetime.now(timezone.utc).isoformat(),
            "event_type":              "reddit_extremism_classification",
            "comment_id":              self.comment_id,
            "subreddit":               self.subreddit,
            "author":                  self.author,
            "risk_label":              self.risk_label,
            "doc_score":               round(self.doc_score, 4),
            "friendly_entity_count":   self.friendly_entity_count,
            "enemy_entity_count":      self.enemy_entity_count,
            "extremist_sentence_count": self.extremist_sentence_count,
            "counter_sentence_count":  self.counter_sentence_count,
            "entity_labels_found":     ",".join(self.entity_labels_found),
            "flagged_sentences":       " | ".join(self.flagged_sentences),
            "comment_snippet":         self.comment_text[:200],
        }


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------

class ExtremistCommentClassifier:
    """
    Classify a Reddit comment for jihadist-extremist content.

    Parameters
    ----------
    use_deberta : bool
        Enable DeBERTa ABSA backend (recommended).
    use_pyabsa : bool
        Enable pyabsa ATEPC backend (corroborating evidence).
    deberta_weight : float
        Weight for DeBERTa in ensemble vote (default 0.6).
    pyabsa_weight : float
        Weight for pyabsa in ensemble vote (default 0.4).
    absa_weight : float
        Contribution of ABSA confidence to per-sentence score (default 0.6).
    toxicity_weight : float
        Contribution of detoxify composite to per-sentence score (default 0.4).
    device : str
        'cuda', 'cpu', or 'auto' (default).
    detoxify_model : str
        'original' (English, fast), 'unbiased', or 'multilingual' (recommended
        for mixed-language Reddit content).
    """

    def __init__(
        self,
        use_deberta:     bool  = True,
        use_pyabsa:      bool  = True,
        deberta_weight:  float = 0.6,
        pyabsa_weight:   float = 0.4,
        absa_weight:     float = 0.6,
        toxicity_weight: float = 0.4,
        device:          str   = "auto",
        detoxify_model:  str   = "multilingual",
    ):
        self._absa_weight     = absa_weight
        self._toxicity_weight = toxicity_weight
        self._detoxify_model_name = detoxify_model
        self._device = device

        # Lazy-loaded models
        self._absa      = None
        self._detoxify  = None
        self._loaded    = False

        # Store init params for lazy construction
        self._init_kwargs = dict(
            use_deberta    = use_deberta,
            use_pyabsa     = use_pyabsa,
            deberta_weight = deberta_weight,
            pyabsa_weight  = pyabsa_weight,
            device         = device,
        )

        # Ensure NLTK punkt tokenizer is available
        self._ensure_nltk()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(
        self,
        comment: str,
        comment_id: str | None = None,
        subreddit:  str | None = None,
        author:     str | None = None,
    ) -> ClassificationResult:
        """Classify a single Reddit comment."""
        self._ensure_loaded()

        comment = comment.strip()
        if not comment:
            return ClassificationResult(
                comment_id=comment_id,
                subreddit=subreddit,
                author=author,
                comment_text=comment,
            )

        sentences       = self._split_sentences(comment)
        sentence_results: list[SentenceResult] = []

        from .entities import iter_entity_matches

        for idx, sent in enumerate(sentences):
            sent = sent.strip()
            if not sent:
                continue

            matches = list(iter_entity_matches(sent))
            if not matches:
                continue

            tox = self._run_detoxify(sent)
            tox_composite = self._toxicity_composite(tox)

            for match in matches:
                absa_out = self._absa.analyze(sent, match.matched_text)
                signal   = self._signal_type(match.entity_type.value, absa_out["sentiment"])
                score    = self._sentence_score(
                    absa_out["confidence"], tox_composite, signal
                )
                sentence_results.append(
                    SentenceResult(
                        sentence               = sent,
                        sentence_index         = idx,
                        entity_label           = match.canonical,
                        entity_surface         = match.matched_text,
                        entity_type            = match.entity_type.value,
                        absa_sentiment         = absa_out["sentiment"],
                        absa_confidence        = absa_out["confidence"],
                        absa_agreement         = absa_out.get("agreement", False),
                        deberta_result         = absa_out.get("deberta_result"),
                        pyabsa_result          = absa_out.get("pyabsa_result"),
                        toxicity_scores        = tox,
                        toxicity_composite     = tox_composite,
                        signal_type            = signal,
                        sentence_extremism_score = score,
                    )
                )

        # Aggregate
        return self._aggregate(
            comment_text    = comment,
            comment_id      = comment_id,
            subreddit       = subreddit,
            author          = author,
            sentence_results = sentence_results,
        )

    def classify_batch(
        self,
        comments: list[dict] | list[str],
    ) -> list[ClassificationResult]:
        """
        Classify a batch of comments.

        Each item can be either:
          - a plain string (comment body)
          - a dict with keys: "body" (required), "id", "subreddit", "author" (optional)
        """
        results = []
        for item in comments:
            if isinstance(item, str):
                results.append(self.classify(item))
            else:
                results.append(self.classify(
                    comment    = item.get("body", ""),
                    comment_id = item.get("id"),
                    subreddit  = item.get("subreddit"),
                    author     = item.get("author"),
                ))
        return results

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _signal_type(entity_type: str, sentiment: str) -> str:
        if entity_type == "friendly":
            if sentiment == "positive":
                return EXTREMIST_SIGNAL
            if sentiment == "negative":
                return COUNTER_SIGNAL
        elif entity_type == "enemy":
            if sentiment == "negative":
                return EXTREMIST_SIGNAL
            if sentiment == "positive":
                return COUNTER_SIGNAL
        return WEAK_SIGNAL

    def _sentence_score(
        self,
        absa_confidence:  float,
        tox_composite:    float,
        signal_type:      str,
    ) -> float:
        multiplier = _SIGNAL_MULTIPLIER[signal_type]
        raw = (
            self._absa_weight     * absa_confidence
            + self._toxicity_weight * tox_composite
        ) * multiplier
        return round(min(raw, 1.0), 4)

    @staticmethod
    def _toxicity_composite(tox: dict) -> float:
        keys = ("toxicity", "severe_toxicity", "identity_attack", "threat")
        vals = [tox.get(k, 0.0) for k in keys]
        return sum(vals) / len(vals)

    @staticmethod
    def _risk_label(score: float) -> str:
        if score >= 0.75:
            return "HIGH"
        if score >= 0.45:
            return "MEDIUM"
        if score >= 0.15:
            return "LOW"
        return "NONE"

    def _aggregate(
        self,
        comment_text:     str,
        comment_id:       str | None,
        subreddit:        str | None,
        author:           str | None,
        sentence_results: list[SentenceResult],
    ) -> ClassificationResult:
        if not sentence_results:
            return ClassificationResult(
                comment_id   = comment_id,
                subreddit    = subreddit,
                author       = author,
                comment_text = comment_text,
            )

        scores    = [s.sentence_extremism_score for s in sentence_results]
        doc_score = round(min(sum(scores) / len(scores), 1.0), 4)

        entity_labels   = list(dict.fromkeys(
            s.entity_label for s in sentence_results
        ))
        friendly_count  = sum(1 for s in sentence_results if s.entity_type == "friendly")
        enemy_count     = sum(1 for s in sentence_results if s.entity_type == "enemy")
        extremist_count = sum(1 for s in sentence_results if s.signal_type == EXTREMIST_SIGNAL)
        counter_count   = sum(1 for s in sentence_results if s.signal_type == COUNTER_SIGNAL)
        flagged_sents   = list(dict.fromkeys(
            s.sentence for s in sentence_results if s.signal_type == EXTREMIST_SIGNAL
        ))

        return ClassificationResult(
            comment_id               = comment_id,
            subreddit                = subreddit,
            author                   = author,
            comment_text             = comment_text,
            sentences                = sentence_results,
            doc_score                = doc_score,
            risk_label               = self._risk_label(doc_score),
            entity_labels_found      = entity_labels,
            friendly_entity_count    = friendly_count,
            enemy_entity_count       = enemy_count,
            extremist_sentence_count = extremist_count,
            counter_sentence_count   = counter_count,
            flagged_sentences        = flagged_sents,
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True

        from detoxify import Detoxify
        from .absa import ABSAEnsemble

        self._absa     = ABSAEnsemble(**self._init_kwargs)
        self._detoxify = Detoxify(self._detoxify_model_name)

    @staticmethod
    def _ensure_nltk() -> None:
        import nltk
        for resource in ("tokenizers/punkt", "tokenizers/punkt_tab"):
            try:
                nltk.data.find(resource)
                return
            except LookupError:
                pass
        nltk.download("punkt",     quiet=True)
        nltk.download("punkt_tab", quiet=True)

    def _split_sentences(self, text: str) -> list[str]:
        from nltk.tokenize import sent_tokenize
        return sent_tokenize(text)

    def _run_detoxify(self, text: str) -> dict:
        try:
            return {k: float(v) for k, v in self._detoxify.predict(text).items()}
        except Exception as exc:
            logger.error("detoxify error: %s", exc)
            return {
                "toxicity": 0.0, "severe_toxicity": 0.0,
                "obscene": 0.0,  "identity_attack": 0.0,
                "insult":  0.0,  "threat": 0.0,
            }
