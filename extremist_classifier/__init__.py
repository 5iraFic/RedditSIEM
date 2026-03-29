"""
extremist_classifier
====================

Islamic-extremist / jihadist comment classifier for Reddit SIEM.

Combines:
  • Entity detection  (FRIENDLY / ENEMY entity lists)
  • ABSA ensemble     (DeBERTa ABSA + pyabsa ATEPC, weighted vote)
  • detoxify          (toxicity, threat, identity_attack, severe_toxicity)

Quickstart
----------
>>> from extremist_classifier import ExtremistCommentClassifier
>>> clf = ExtremistCommentClassifier()
>>> result = clf.classify(
...     "We must support the mujahideen in their holy war!",
...     comment_id="t1_abc",
...     subreddit="r/example",
... )
>>> print(result.risk_label, result.doc_score)
"""

__version__ = "0.1.0"

from .absa import ABSAEnsemble, DeBERTaABSAAnalyzer, FallbackSentimentAnalyzer, PyABSAAnalyzer
from .classifier import (
    ClassificationResult,
    ExtremistCommentClassifier,
    SentenceResult,
)
from .entities import ENEMY_ENTITIES, FRIENDLY_ENTITIES, EntityMatch, EntityType

__all__ = [
    # Main interface
    "ExtremistCommentClassifier",
    "ClassificationResult",
    "SentenceResult",
    # ABSA backends (for standalone use)
    "ABSAEnsemble",
    "DeBERTaABSAAnalyzer",
    "PyABSAAnalyzer",
    "FallbackSentimentAnalyzer",
    # Entity utilities
    "EntityType",
    "EntityMatch",
    "FRIENDLY_ENTITIES",
    "ENEMY_ENTITIES",
]
