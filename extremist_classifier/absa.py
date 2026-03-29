"""
ABSA (Aspect-Based Sentiment Analysis) wrappers.

Three backends, unified interface:

  1. DeBERTaABSAAnalyzer   – yangheng/deberta-v3-base-absa-v1.1
       Input: "[CLS] sentence [SEP] aspect [SEP]"
       Labels: Negative (0), Neutral (1), Positive (2)
       Weight in ensemble: 0.6  (dedicated ABSA model, higher capacity)

  2. PyABSAAnalyzer         – pyabsa ATEPC multilingual checkpoint
       Aspect-Term Extraction + Polarity Classification.
       The aspect (entity surface form) is injected; the model returns
       polarity over (Negative, Neutral, Positive).
       Weight in ensemble: 0.4

  3. FallbackSentimentAnalyzer – cardiffnlp/twitter-roberta-base-sentiment-latest
       Sentence-level only (no aspect injection); used when both primary
       models are unavailable.

ABSAEnsemble runs whatever backends are available and merges results via
a soft weighted vote:
  - weighted score per class = sum(weight_i * prob_i)
  - winning class = argmax of weighted scores
  - confidence = winning class weighted score / sum of all weighted scores
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Canonical sentiment labels used everywhere
POSITIVE = "positive"
NEGATIVE = "negative"
NEUTRAL  = "neutral"
LABELS   = (NEGATIVE, NEUTRAL, POSITIVE)   # matches typical model logit order


# ---------------------------------------------------------------------------
# DeBERTa ABSA backend
# ---------------------------------------------------------------------------

class DeBERTaABSAAnalyzer:
    """
    Wraps yangheng/deberta-v3-base-absa-v1.1.

    Input format expected by the model:
        "[CLS] <sentence> [SEP] <aspect> [SEP]"
    which the HuggingFace tokenizer builds automatically when you pass
    text_pair=(sentence, aspect).

    Label mapping (from model config):
        0 → Negative, 1 → Neutral, 2 → Positive
    """

    MODEL_ID = "yangheng/deberta-v3-base-absa-v1.1"
    LABEL_MAP = {0: NEGATIVE, 1: NEUTRAL, 2: POSITIVE}

    def __init__(self, device: str = "auto"):
        self.device = self._resolve_device(device)
        self._tokenizer = None
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.MODEL_ID
            ).to(self.device)
            self._model.eval()
            logger.info("DeBERTa ABSA loaded (%s)", self.MODEL_ID)
        except Exception as exc:
            logger.warning("DeBERTa ABSA not available: %s", exc)

    def analyze(self, sentence: str, aspect: str) -> dict | None:
        """
        Returns:
          {
            "sentiment":   "positive" | "negative" | "neutral",
            "confidence":  float,
            "probs":       {"negative": float, "neutral": float, "positive": float},
            "method":      "deberta_absa",
          }
        Returns None if the model failed to load.
        """
        self._load()
        if self._model is None:
            return None

        try:
            import torch
            import torch.nn.functional as F

            # Truncate sentence to keep the full aspect visible
            sentence = sentence[:400]

            enc = self._tokenizer(
                sentence,
                aspect,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            ).to(self.device)

            with torch.no_grad():
                logits = self._model(**enc).logits
            probs = F.softmax(logits, dim=-1)[0].cpu().tolist()

            probs_dict = {
                NEGATIVE: probs[0],
                NEUTRAL:  probs[1],
                POSITIVE: probs[2],
            }
            best_label = max(probs_dict, key=probs_dict.__getitem__)
            return {
                "sentiment": best_label,
                "confidence": probs_dict[best_label],
                "probs": probs_dict,
                "method": "deberta_absa",
            }
        except Exception as exc:
            logger.error("DeBERTa inference error: %s", exc)
            return None

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"


# ---------------------------------------------------------------------------
# PyABSA backend
# ---------------------------------------------------------------------------

class PyABSAAnalyzer:
    """
    Wraps pyabsa's ATEPC (Aspect-Term Extraction + Polarity Classification)
    multilingual checkpoint.

    The entity surface form IS guaranteed to be present in the sentence
    (we extracted that sentence because it contains the entity), so ATEPC
    injection via the `aspects` parameter is always valid.
    """

    def __init__(self):
        self._model = None
        self._tried = False   # prevent repeated load attempts on failure

    def _load(self) -> None:
        if self._model is not None or self._tried:
            return
        self._tried = True
        try:
            # pyabsa 2.x API
            from pyabsa import ATEPCCheckpoint
            from pyabsa import AspectTermExtraction as ATEPC

            self._model = ATEPC.AspectExtractor(
                ATEPCCheckpoint.MULTILINGUAL,
                auto_device=True,
            )
            logger.info("pyabsa ATEPC (multilingual) loaded")
        except ImportError:
            # pyabsa 3.x changed the public API — try alternative import
            try:
                from pyabsa.tasks.AspectTermExtraction import AspectExtractor
                from pyabsa import available_checkpoints

                self._model = AspectExtractor("multilingual", auto_device=True)
                logger.info("pyabsa ATEPC loaded via 3.x API")
            except Exception as exc2:
                logger.warning("pyabsa not available (tried 2.x and 3.x API): %s", exc2)
                return
        except Exception as exc:
            logger.warning("pyabsa not available: %s", exc)

    def analyze(self, sentence: str, aspect: str) -> dict | None:
        """
        Returns:
          {
            "sentiment":   "positive" | "negative" | "neutral",
            "confidence":  float,
            "probs":       {"negative": float, "neutral": float, "positive": float},
            "method":      "pyabsa",
          }
        Returns None if pyabsa failed to load or the aspect was not found.
        """
        self._load()
        if self._model is None:
            return None

        try:
            results = self._model.extract_aspect(
                inference_source=[sentence],
                pred_sentiment=True,
            )

            if not results or not results[0].get("aspect"):
                return None

            r = results[0]
            aspects_lower = [a.lower() for a in r["aspect"]]
            aspect_lower  = aspect.lower()

            # Find the index of our target aspect (exact or substring match)
            idx = None
            for i, a in enumerate(aspects_lower):
                if aspect_lower in a or a in aspect_lower:
                    idx = i
                    break

            if idx is None:
                return None  # pyabsa didn't extract our entity

            sentiment = (r.get("sentiment") or [])[idx].lower()
            probs_raw = (r.get("probs") or [])[idx]

            if isinstance(probs_raw, (list, tuple)) and len(probs_raw) == 3:
                probs_dict = {
                    NEGATIVE: float(probs_raw[0]),
                    NEUTRAL:  float(probs_raw[1]),
                    POSITIVE: float(probs_raw[2]),
                }
            else:
                # Fallback: assign full confidence to reported label
                probs_dict = {NEGATIVE: 0.0, NEUTRAL: 0.0, POSITIVE: 0.0}
                if sentiment in probs_dict:
                    probs_dict[sentiment] = 1.0

            best_label = max(probs_dict, key=probs_dict.__getitem__)
            return {
                "sentiment":  best_label,
                "confidence": probs_dict[best_label],
                "probs":      probs_dict,
                "method":     "pyabsa",
            }
        except Exception as exc:
            logger.error("pyabsa inference error: %s", exc)
            return None


# ---------------------------------------------------------------------------
# Sentence-level fallback
# ---------------------------------------------------------------------------

class FallbackSentimentAnalyzer:
    """
    cardiffnlp/twitter-roberta-base-sentiment-latest.
    No aspect injection; sentence-level only.
    Used when both primary ABSA models are unavailable.
    """

    MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    LABEL_MAP = {
        "positive": POSITIVE, "pos": POSITIVE, "label_2": POSITIVE,
        "negative": NEGATIVE, "neg": NEGATIVE, "label_0": NEGATIVE,
        "neutral":  NEUTRAL,  "neu": NEUTRAL,  "label_1": NEUTRAL,
    }

    def __init__(self, device: str = "auto"):
        self.device = DeBERTaABSAAnalyzer._resolve_device(device)
        self._pipe = None

    def _load(self) -> None:
        if self._pipe is not None:
            return
        try:
            from transformers import pipeline as hf_pipeline

            self._pipe = hf_pipeline(
                "sentiment-analysis",
                model=self.MODEL_ID,
                device=0 if self.device == "cuda" else -1,
                truncation=True,
                max_length=512,
            )
            logger.info("Fallback sentiment model loaded (%s)", self.MODEL_ID)
        except Exception as exc:
            logger.warning("Fallback sentiment model not available: %s", exc)

    def analyze(self, sentence: str) -> dict | None:
        self._load()
        if self._pipe is None:
            return None
        try:
            result = self._pipe(sentence)[0]
            label = self.LABEL_MAP.get(result["label"].lower(), NEUTRAL)
            conf  = float(result["score"])
            probs = {NEGATIVE: 0.0, NEUTRAL: 0.0, POSITIVE: 0.0}
            probs[label] = conf
            return {
                "sentiment":  label,
                "confidence": conf,
                "probs":      probs,
                "method":     "twitter_roberta",
            }
        except Exception as exc:
            logger.error("Fallback sentiment error: %s", exc)
            return None


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

class ABSAEnsemble:
    """
    Runs DeBERTa ABSA and/or pyabsa and merges via soft weighted vote.

    Weighted vote per class:
        score_c = sum_i(weight_i * prob_i_c)
    Winning class:
        argmax(score_c)
    Confidence:
        score_winning / sum(score_c)   (normalised so it's always in [0,1])

    Falls back to FallbackSentimentAnalyzer if neither primary model loads.
    """

    def __init__(
        self,
        use_deberta: bool = True,
        use_pyabsa:  bool = True,
        deberta_weight: float = 0.6,
        pyabsa_weight:  float = 0.4,
        device: str = "auto",
    ):
        self._deberta  = DeBERTaABSAAnalyzer(device=device) if use_deberta else None
        self._pyabsa   = PyABSAAnalyzer()                    if use_pyabsa  else None
        self._fallback = FallbackSentimentAnalyzer(device=device)
        self._dw = deberta_weight
        self._pw = pyabsa_weight

    def analyze(self, sentence: str, aspect: str) -> dict:
        """
        Returns:
          {
            "sentiment":       "positive" | "negative" | "neutral",
            "confidence":      float,
            "agreement":       bool,          # True if both models agreed
            "deberta_result":  dict | None,
            "pyabsa_result":   dict | None,
            "method":          str,
          }
        Never raises; always returns a valid dict.
        """
        deberta_r = self._deberta.analyze(sentence, aspect) if self._deberta else None
        pyabsa_r  = self._pyabsa.analyze(sentence, aspect)  if self._pyabsa  else None

        results_with_weights = [
            (deberta_r, self._dw),
            (pyabsa_r,  self._pw),
        ]
        active = [(r, w) for r, w in results_with_weights if r is not None]

        if not active:
            # Both primary models failed; use fallback
            fb = self._fallback.analyze(sentence)
            if fb is None:
                fb = {"sentiment": NEUTRAL, "confidence": 0.5,
                      "probs": {NEGATIVE: 0.33, NEUTRAL: 0.34, POSITIVE: 0.33},
                      "method": "default"}
            return {
                "sentiment":      fb["sentiment"],
                "confidence":     fb["confidence"],
                "agreement":      False,
                "deberta_result": None,
                "pyabsa_result":  None,
                "method":         fb["method"],
            }

        # Soft weighted vote over probability distributions
        weighted = {NEGATIVE: 0.0, NEUTRAL: 0.0, POSITIVE: 0.0}
        total_weight = sum(w for _, w in active)

        for result, weight in active:
            norm_w = weight / total_weight
            for label in LABELS:
                weighted[label] += norm_w * result["probs"].get(label, 0.0)

        best = max(weighted, key=weighted.__getitem__)
        total = sum(weighted.values()) or 1.0
        confidence = weighted[best] / total

        sentiments = [r["sentiment"] for r, _ in active]
        agreement  = len(set(sentiments)) == 1

        methods = "+".join(r["method"] for r, _ in active)
        return {
            "sentiment":      best,
            "confidence":     round(confidence, 4),
            "agreement":      agreement,
            "deberta_result": deberta_r,
            "pyabsa_result":  pyabsa_r,
            "method":         methods,
        }
