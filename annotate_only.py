"""
annotate_only.py — Run only the teacher annotation step.

Useful when you want to do annotation in a separate session from training,
or when you need to split annotation across multiple days due to API rate limits.

Usage:
  $env:GROQ_API_KEY = "gsk_..."
  python annotate_only.py
"""

import os
import sys
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    print("\n" + "█" * 60)
    print("  P6: TEACHER ANNOTATION (Groq / Llama 3.3 70B)")
    print("█" * 60 + "\n")

    # ── 1. Load data ──────────────────────────────────────────────────────
    print("► Loading FinancialPhraseBank...")
    from src.data.loader import FinancialPhraseBankLoader

    loader = FinancialPhraseBankLoader(seed=42)
    split = loader.load()
    print(f"  Train: {len(split.train)} samples")
    print(f"  Val:   {len(split.val)} samples")
    print(f"  Test:  {len(split.test)} samples (NOT annotated)")

    # ── 2. Set up the teacher ─────────────────────────────────────────────
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        print("\n✗ GROQ_API_KEY environment variable not set.")
        print('  Run: $env:GROQ_API_KEY = "gsk_..."')
        sys.exit(1)

    cache_path = "outputs/teacher_cache_groq.json"
    from src.models.teacher import GroqTeacher

    teacher = GroqTeacher(api_key=groq_key, cache_path=cache_path)

    # ── 3. Annotate ───────────────────────────────────────────────────────
    all_to_annotate = split.train + split.val
    cost = teacher.estimate_cost(len(all_to_annotate))

    print(f"\n► Annotation plan:")
    print(f"  Total samples (train + val): {len(all_to_annotate)}")
    print(f"  Already cached:              {cost['cached']}")
    print(f"  New API calls needed:        {cost['new_api_calls']}")
    print(f"  Estimated time:              ~{cost['estimated_time_minutes']} min")
    print(f"  Cost:                        $0.00 (free tier)")
    if cost['new_api_calls'] > 1000:
        print(f"  ⚠ {cost['daily_quota_warning']}")

    if cost['new_api_calls'] == 0:
        print("\n✓ All samples already annotated! Nothing to do.")
        return

    print(f"\n► Starting annotation. Progress prints every 50 samples.")
    print(f"  Cache file: {cache_path}")
    print(f"  You can interrupt with Ctrl+C; progress is saved.\n")

    teacher.annotate_batch(all_to_annotate, verbose=True)

    # ── 4. Summary ────────────────────────────────────────────────────────
    print("\n► Annotation summary:")
    annotated_count = sum(1 for s in all_to_annotate if s.soft_labels is not None)
    remaining = len(all_to_annotate) - annotated_count
    print(f"  Annotated: {annotated_count}/{len(all_to_annotate)}")
    print(f"  Remaining: {remaining}")

    if remaining > 0:
        print(f"\n  → Re-run this script tomorrow to continue annotating the remaining {remaining}.")
    else:
        print(f"\n  ✓ Annotation complete! Ready to train.")

    # Teacher agreement stats (on whatever subset was annotated)
    from src.evaluation.metrics import compute_teacher_agreement

    annotated_train = [s for s in split.train if s.soft_labels is not None]
    if annotated_train:
        agree = compute_teacher_agreement(annotated_train)
        print(f"\n► Teacher quality (on {len(annotated_train)} annotated train samples):")
        print(f"  Teacher-label agreement: {agree.get('teacher_hard_agreement', 'N/A')}")
        print(f"  Mean teacher confidence: {agree.get('mean_confidence', 'N/A')}")
        print(f"  High-confidence ratio:   {agree.get('high_confidence_ratio', 'N/A')}")


if __name__ == "__main__":
    main()
