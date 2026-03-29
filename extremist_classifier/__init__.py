"""
extremist_classifier
====================

Islamic-extremist / jihadist comment classifier for Reddit SIEM.

Combines:
  • Entity detection  (FRIENDLY / ENEMY entity lists)
  • ABSA             (Aspect-Based Sentiment Analysis via pyabsa / DeBERTa)
  • detoxify         (toxicity, threat, identity_attack scores)

Quickstart
----------
>>> from extremist_classifier import ExtremistCommentClassifier
>>> clf = ExtremistCommentClassifier()
>>> result = clf.classify("We must support the mujahideen in their holy war!")
>>> print(result.label, result.score)
"""

from .classifier import ClassificationResult, EvidenceItem, ExtremistCommentClassifier
from .entities import ENEMY_ENTITIES, FRIENDLY_ENTITIES, EntityMatch, EntityType

__all__ = [
    "ExtremistCommentClassifier",
    "ClassificationResult",
    "EvidenceItem",
    "EntityType",
    "EntityMatch",
    "FRIENDLY_ENTITIES",
    "ENEMY_ENTITIES",
]
