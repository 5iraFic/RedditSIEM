"""
Usage examples for ExtremistCommentClassifier.

Run:
    pip install -r requirements.txt
    python example.py
"""

import json
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

from extremist_classifier import ExtremistCommentClassifier

# ---------------------------------------------------------------------------
# Sample comments (synthetic – for testing only)
# ---------------------------------------------------------------------------

SAMPLES = [
    # (comment, expected_risk)
    (
        "We must support the mujahideen in their jihad against the kuffar. "
        "May Allah grant them victory!",
        "HIGH",
    ),
    (
        "ISIS is doing the right thing by establishing the caliphate. "
        "Death to all infidels and Jews.",
        "HIGH",
    ),
    (
        "The shaheed who carried out the operation is a true martyr. "
        "We should all pledge bay'ah to the caliph.",
        "HIGH",
    ),
    (
        "People say ISIS is brutal but they forget what the Americans did in Iraq.",
        "LOW",
    ),
    (
        "The mujahideen have fought for decades in Afghanistan.",
        "LOW",
    ),
    (
        "ISIS must be stopped. Their violence against civilians is unacceptable.",
        "NONE",  # counter-signal: negative sentiment toward friendly entity
    ),
    (
        "I visited the mosque last Friday. The sermon was about patience and kindness.",
        "NONE",
    ),
    (
        "Just made some amazing shawarma for dinner. Highly recommend the recipe!",
        "NONE",
    ),
]


def main() -> None:
    print("Loading classifier (models download on first run — may take a few minutes)...\n")

    clf = ExtremistCommentClassifier(
        use_deberta     = True,
        use_pyabsa      = True,
        deberta_weight  = 0.6,
        pyabsa_weight   = 0.4,
        absa_weight     = 0.6,
        toxicity_weight = 0.4,
        detoxify_model  = "multilingual",
        device          = "auto",
    )

    # ----- Single comment classification -----
    print("=" * 70)
    print("Single comment demo")
    print("=" * 70)

    result = clf.classify(
        comment    = SAMPLES[0][0],
        comment_id = "t1_demo001",
        subreddit  = "r/example",
        author     = "u/testuser",
    )
    print(result.summary())
    print()

    # SIEM event output
    siem = result.to_siem_event()
    print("SIEM event JSON:")
    print(json.dumps(siem, indent=2))
    print()

    # ----- Batch classification -----
    print("=" * 70)
    print("Batch evaluation")
    print("=" * 70)

    batch = [
        {"body": text, "id": f"t1_{i:03d}", "subreddit": "r/test", "author": "u/x"}
        for i, (text, _) in enumerate(SAMPLES)
    ]
    results = clf.classify_batch(batch)

    correct = 0
    for (_, expected), res in zip(SAMPLES, results):
        ok = "✓" if res.risk_label == expected else "✗"
        if res.risk_label == expected:
            correct += 1
        print(
            f"{ok} [{res.risk_label:6s}] expected={expected:6s}  "
            f"score={res.doc_score:.3f}  entities={res.entity_labels_found}"
        )

    print(f"\nExact match: {correct}/{len(SAMPLES)}")

    # ----- Reddit stream integration pattern -----
    print()
    print("=" * 70)
    print("Reddit stream integration pattern")
    print("=" * 70)
    print("""
# Typical SIEM ingestion loop:

import praw
from extremist_classifier import ExtremistCommentClassifier

clf = ExtremistCommentClassifier(detoxify_model="multilingual")

reddit = praw.Reddit(...)
for comment in reddit.subreddit("all").stream.comments(skip_existing=True):
    result = clf.classify(
        comment    = comment.body,
        comment_id = comment.id,
        subreddit  = str(comment.subreddit),
        author     = str(comment.author),
    )
    if result.risk_label in ("HIGH", "MEDIUM"):
        event = result.to_siem_event()
        # ship to Elasticsearch / Splunk / webhook
        print(f"[ALERT] {result.risk_label} | {comment.id} | score={result.doc_score:.3f}")
    """)


if __name__ == "__main__":
    main()
