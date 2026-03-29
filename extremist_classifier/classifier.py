"""
ExtremistCommentClassifier  –  main public interface.

Pipeline per comment:
  1. Split text into sentences (spaCy).
  2. For each sentence: scan entity lists (entities.py).
  3. For each (sentence, entity) pair:
       a. ABSA  → sentiment toward the entity (absa.py)
       b. detoxify → per-sentence toxicity scores
  4. Compute an extremism sub-score for each evidence item.
  5. Aggregate → final score + label + structured evidence.

Scoring rationale:
  • FRIENDLY entity + POSITIVE sentiment  → strong extremist signal
  • ENEMY entity   + NEGATIVE sentiment  → strong extremist signal
  • Any entity context + high toxicity/threat/identity_attack → amplifier
  • FRIENDLY + NEGATIVE (denouncing) → weak signal (mention only)
  • pyabsa auto-extracted aspects that match entity lists add corroborating
    evidence independently of the forced-aspect analysis above.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class EvidenceItem:
    entity: str
    entity_type: str                    # "friendly" | "enemy"
    matched_text: str
    sentence: str
    absa: dict                          # {"sentiment", "confidence", "method"}
    detoxify: dict                      # raw detoxify scores
    sub_score: float                    # [0, 1]


@dataclass
class ClassificationResult:
    text: str
    score: float                        # [0, 1]  aggregated extremism score
    label: str                          # "extremist" | "suspicious" | "borderline" | "clean"
    is_extremist: bool
    evidence: list[EvidenceItem] = field(default_factory=list)
    # pyabsa auto-detected aspects (may corroborate evidence)
    auto_aspects: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Label   : {self.label} (score={self.score:.3f})",
            f"Entities: {len(self.evidence)} mention(s)",
        ]
        for ev in self.evidence:
            lines.append(
                f"  [{ev.entity_type.upper()}] {ev.entity!r}  "
                f"→ sentiment={ev.absa['sentiment']} "
                f"(conf={ev.absa['confidence']:.2f})  "
                f"tox={ev.detoxify.get('toxicity', 0):.2f}  "
                f"sub_score={ev.sub_score:.3f}"
            )
            lines.append(f"    sentence: {ev.sentence[:120]!r}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------

class ExtremistCommentClassifier:
    """
    Classify a Reddit comment (or any short text) for jihadist-extremist content.

    Parameters
    ----------
    detoxify_model : str
        'original' (English-only, fastest) or 'multilingual' (recommended for
        mixed-language Reddit).
    device : str | None
        'cuda', 'cpu', or None (auto-detect).
    score_threshold : float
        Score above which is_extremist = True.  Default 0.45.
    """

    SCORE_THRESHOLD = 0.45

    def __init__(
        self,
        detoxify_model: str = "multilingual",
        device: Optional[str] = None,
        score_threshold: float = SCORE_THRESHOLD,
    ):
        import torch

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.score_threshold = score_threshold
        self._nlp = None
        self._detoxify = None
        self._absa = None
        self._detoxify_model_name = detoxify_model

    # ------------------------------------------------------------------
    # Lazy initialization
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._nlp is not None:
            return

        import spacy
        from detoxify import Detoxify

        from .absa import ABSAAnalyzer

        try:
            self._nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning(
                "spaCy model 'en_core_web_sm' not found. "
                "Run: python -m spacy download en_core_web_sm"
            )
            # Minimal fallback: sentence split on ". "
            self._nlp = None

        self._detoxify = Detoxify(self._detoxify_model_name)
        self._absa = ABSAAnalyzer(device=self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, text: str) -> ClassificationResult:
        """
        Classify a single comment.

        Returns a ClassificationResult with score, label, and detailed evidence.
        """
        self._load()

        text = text.strip()
        if not text:
            return ClassificationResult(
                text=text, score=0.0, label="clean", is_extremist=False
            )

        sentences = self._split_sentences(text)

        evidence: list[EvidenceItem] = []
        auto_aspects: list[dict] = []

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue

            # --- Entity matching ---
            from .entities import iter_entity_matches

            matches = list(iter_entity_matches(sent))

            if not matches:
                continue  # no tracked entity in this sentence

            # --- detoxify on the sentence ---
            tox = self._run_detoxify(sent)

            # --- pyabsa auto-extraction (corroborating evidence) ---
            auto_asp = self._absa.extract_aspects(sent)
            auto_aspects.extend(
                {**a, "sentence": sent} for a in auto_asp
            )

            # --- Forced ABSA per entity mention ---
            for match in matches:
                absa_result = self._absa.analyze(sent, match.canonical)
                sub = self._compute_sub_score(
                    match.entity_type.value, absa_result, tox
                )
                evidence.append(
                    EvidenceItem(
                        entity=match.canonical,
                        entity_type=match.entity_type.value,
                        matched_text=match.matched_text,
                        sentence=sent,
                        absa=absa_result,
                        detoxify=tox,
                        sub_score=sub,
                    )
                )

        # --- No entity mentions: run detoxify on full text as fallback ---
        if not evidence:
            tox = self._run_detoxify(text)
            fallback_score = self._toxicity_score(tox) * 0.5
            label = self._label(fallback_score)
            return ClassificationResult(
                text=text,
                score=round(fallback_score, 4),
                label=label,
                is_extremist=fallback_score >= self.score_threshold,
                evidence=[],
                auto_aspects=auto_aspects,
            )

        # --- Aggregate ---
        scores = [ev.sub_score for ev in evidence]
        max_score = max(scores)
        avg_score = sum(scores) / len(scores)
        final_score = round(0.65 * max_score + 0.35 * avg_score, 4)
        label = self._label(final_score)

        return ClassificationResult(
            text=text,
            score=final_score,
            label=label,
            is_extremist=final_score >= self.score_threshold,
            evidence=evidence,
            auto_aspects=auto_aspects,
        )

    def classify_batch(self, texts: list[str]) -> list[ClassificationResult]:
        """Classify a list of comments."""
        return [self.classify(t) for t in texts]

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _compute_sub_score(
        self,
        entity_type: str,   # "friendly" | "enemy"
        absa: dict,
        tox: dict,
    ) -> float:
        sentiment = absa["sentiment"]
        confidence = absa["confidence"]
        score = 0.0

        if entity_type == "friendly":
            if sentiment == "positive":
                # Glorifying IS / mujahideen / martyrdom → strong signal
                score += confidence * 0.70
            elif sentiment == "neutral":
                # Neutral mention still worth noting
                score += confidence * 0.10
            # Negative toward friendly entities → counter-narrative, not extremist
        elif entity_type == "enemy":
            if sentiment == "negative":
                # Expressing hatred toward kuffar / Jews / Americans → strong signal
                score += confidence * 0.65
            elif sentiment == "neutral":
                score += confidence * 0.08

        # Toxicity amplifier (weighted average of threat + identity_attack + toxicity)
        tox_composite = (
            tox.get("toxicity", 0) * 0.25
            + tox.get("identity_attack", 0) * 0.40
            + tox.get("threat", 0) * 0.35
        )
        score += tox_composite * 0.35

        return min(score, 1.0)

    @staticmethod
    def _toxicity_score(tox: dict) -> float:
        return (
            tox.get("toxicity", 0) * 0.25
            + tox.get("identity_attack", 0) * 0.40
            + tox.get("threat", 0) * 0.35
        )

    @staticmethod
    def _label(score: float) -> str:
        if score >= 0.70:
            return "extremist"
        if score >= 0.45:
            return "suspicious"
        if score >= 0.25:
            return "borderline"
        return "clean"

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _split_sentences(self, text: str) -> list[str]:
        if self._nlp is not None:
            doc = self._nlp(text)
            return [sent.text for sent in doc.sents]
        # Minimal fallback: split on sentence-ending punctuation
        parts = re.split(r"(?<=[.!?])\s+", text)
        return parts if parts else [text]

    def _run_detoxify(self, text: str) -> dict:
        try:
            result = self._detoxify.predict(text)
            return {k: float(v) for k, v in result.items()}
        except Exception as exc:
            logger.error("detoxify error: %s", exc)
            return {
                "toxicity": 0.0,
                "severe_toxicity": 0.0,
                "obscene": 0.0,
                "identity_attack": 0.0,
                "insult": 0.0,
                "threat": 0.0,
            }
