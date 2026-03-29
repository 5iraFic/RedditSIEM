"""
ABSA (Aspect-Based Sentiment Analysis) wrapper.

Strategy (in priority order):
  1. yangheng/deberta-v3-base-absa-v1.1  – dedicated ABSA model that accepts
     the "[aspect]:[sentence]" prompt format and returns Positive/Negative/Neutral.
  2. pyabsa ATEPC  – auto-extracts aspects AND predicts sentiment; results are
     matched against our entity lists by the caller.
  3. cardiffnlp/twitter-roberta-base-sentiment-latest  – sentence-level fallback
     when neither dedicated ABSA model is available.

All methods return a uniform dict:
  {
    "sentiment":  "positive" | "negative" | "neutral",
    "confidence": float,          # [0, 1]
    "method":     str,            # which backend produced the result
  }
"""

from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


class ABSAAnalyzer:
    """
    Lazy-loading ABSA analyzer.  Models are downloaded on first use.
    """

    def __init__(self, device: Optional[str] = None):
        import torch

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._deberta_pipe = None
        self._pyabsa_model = None
        self._fallback_pipe = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, sentence: str, aspect: str) -> dict:
        """
        Return sentiment of *sentence* toward *aspect*.

        Uses the highest-priority available backend.
        """
        self._ensure_loaded()

        if self._deberta_pipe:
            return self._analyze_deberta(sentence, aspect)

        if self._fallback_pipe:
            return self._analyze_fallback(sentence)

        return {"sentiment": "neutral", "confidence": 0.5, "method": "none"}

    def extract_aspects(self, text: str) -> list[dict]:
        """
        Auto-extract aspects and their sentiments from *text* via pyabsa.

        Returns a list of dicts:
          {"aspect": str, "sentiment": str, "confidence": float}

        Falls back to an empty list if pyabsa is unavailable.
        """
        self._ensure_loaded()

        if not self._pyabsa_model:
            return []

        try:
            results = self._pyabsa_model.extract_aspect(
                inference_source=[text],
                pred_sentiment=True,
            )
            aspects = []
            if results and results[0].get("aspect"):
                r = results[0]
                for i, asp in enumerate(r["aspect"]):
                    sentiment = (r.get("sentiment") or [])[i] if r.get("sentiment") else "neutral"
                    probs = (r.get("probs") or [])[i] if r.get("probs") else [0.34, 0.33, 0.33]
                    aspects.append(
                        {
                            "aspect": asp.lower().strip(),
                            "sentiment": sentiment.lower(),
                            "confidence": float(max(probs)),
                        }
                    )
            return aspects
        except Exception as exc:
            logger.error("pyabsa extraction error: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True  # set early to prevent recursive calls
        self._load_deberta()
        self._load_pyabsa()
        self._load_fallback()

    def _load_deberta(self) -> None:
        try:
            from transformers import pipeline as hf_pipeline

            self._deberta_pipe = hf_pipeline(
                "text-classification",
                model="yangheng/deberta-v3-base-absa-v1.1",
                device=0 if self.device == "cuda" else -1,
                truncation=True,
                max_length=512,
            )
            logger.info("DeBERTa ABSA model loaded (yangheng/deberta-v3-base-absa-v1.1)")
        except Exception as exc:
            logger.warning("DeBERTa ABSA not available: %s", exc)

    def _load_pyabsa(self) -> None:
        try:
            from pyabsa import ATEPCCheckpoint
            from pyabsa import AspectTermExtraction as ATEPC

            self._pyabsa_model = ATEPC.AspectExtractor(
                ATEPCCheckpoint.MULTILINGUAL,
                auto_device=True,
            )
            logger.info("pyabsa ATEPC (multilingual) loaded")
        except Exception as exc:
            logger.warning("pyabsa not available: %s", exc)

    def _load_fallback(self) -> None:
        if self._deberta_pipe:
            return  # primary model available; skip fallback
        try:
            from transformers import pipeline as hf_pipeline

            self._fallback_pipe = hf_pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0 if self.device == "cuda" else -1,
                truncation=True,
                max_length=512,
            )
            logger.info("Fallback sentiment model loaded (cardiffnlp/twitter-roberta)")
        except Exception as exc:
            logger.warning("Fallback sentiment model not available: %s", exc)

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------

    _LABEL_MAP_DEBERTA = {
        "positive": "positive",
        "negative": "negative",
        "neutral": "neutral",
    }

    def _analyze_deberta(self, sentence: str, aspect: str) -> dict:
        """DeBERTa ABSA: input format is '[aspect]:[sentence]'."""
        input_text = f"{aspect}:{sentence}"
        try:
            result = self._deberta_pipe(input_text)[0]
            label = self._normalize_label(result["label"])
            return {
                "sentiment": label,
                "confidence": float(result["score"]),
                "method": "deberta_absa",
            }
        except Exception as exc:
            logger.error("DeBERTa ABSA inference error: %s", exc)
            # Fall through to fallback
            if self._fallback_pipe:
                return self._analyze_fallback(sentence)
            return {"sentiment": "neutral", "confidence": 0.5, "method": "error"}

    _LABEL_MAP_TWITTER = {
        "positive": "positive",
        "negative": "negative",
        "neutral": "neutral",
        "label_0": "negative",
        "label_1": "neutral",
        "label_2": "positive",
    }

    def _analyze_fallback(self, sentence: str) -> dict:
        try:
            result = self._fallback_pipe(sentence)[0]
            label = self._normalize_label(result["label"])
            return {
                "sentiment": label,
                "confidence": float(result["score"]),
                "method": "twitter_roberta",
            }
        except Exception as exc:
            logger.error("Fallback sentiment error: %s", exc)
            return {"sentiment": "neutral", "confidence": 0.5, "method": "error"}

    @staticmethod
    def _normalize_label(label: str) -> str:
        label = label.lower().strip()
        mapping = {
            "positive": "positive",
            "pos": "positive",
            "label_2": "positive",
            "negative": "negative",
            "neg": "negative",
            "label_0": "negative",
            "neutral": "neutral",
            "neu": "neutral",
            "label_1": "neutral",
        }
        return mapping.get(label, "neutral")
