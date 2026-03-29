"""
Usage examples for ExtremistCommentClassifier.

Run:
    python example.py
"""

import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

from extremist_classifier import ExtremistCommentClassifier

# ---------------------------------------------------------------------------
# Sample comments  (synthetic – for testing only)
# ---------------------------------------------------------------------------

SAMPLES = [
    # --- Expected: extremist ---
    (
        "We must support the mujahideen in their jihad against the kuffar. "
        "May Allah grant them victory!",
        "extremist",
    ),
    (
        "ISIS is doing the right thing by establishing the caliphate. "
        "Death to all infidels and Jews.",
        "extremist",
    ),
    (
        "The shaheed who blew himself up is a true martyr. "
        "We should all follow Baghdadi's example.",
        "extremist",
    ),
    # --- Expected: suspicious / borderline ---
    (
        "People say ISIS is brutal but they forget what the Americans did in Iraq.",
        "borderline",
    ),
    (
        "The mujahideen have been fighting for decades in Afghanistan.",
        "borderline",
    ),
    # --- Expected: clean ---
    (
        "I visited the mosque last Friday. The sermon was about patience and kindness.",
        "clean",
    ),
    (
        "Just made some amazing shawarma for dinner. Highly recommend the recipe!",
        "clean",
    ),
    (
        "The documentary about the Crusades was really educational.",
        "clean",
    ),
]


def main() -> None:
    print("Loading classifier (this may take a minute on first run – models download)...\n")
    clf = ExtremistCommentClassifier(detoxify_model="multilingual")

    correct = 0
    for text, expected in SAMPLES:
        result = clf.classify(text)
        match = "✓" if result.label in (expected, "suspicious") else "✗"
        print(f"{match} [{result.label:12s}] expected={expected:12s}  score={result.score:.3f}")
        print(f"  TEXT: {text[:90]!r}")
        if result.evidence:
            for ev in result.evidence:
                print(
                    f"  → [{ev.entity_type}] {ev.entity!r}  "
                    f"sentiment={ev.absa['sentiment']}  "
                    f"tox={ev.detoxify.get('toxicity', 0):.2f}"
                )
        print()
        if result.label == expected:
            correct += 1

    print(f"Exact match: {correct}/{len(SAMPLES)}")


if __name__ == "__main__":
    main()
