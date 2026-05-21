"""
annotate_subset.py — Smart top-up annotation for stratified subset approach.

Audits existing annotations, then annotates the smallest-classes-first
until we hit a target stratified subset size.

Usage:
  $env:GROQ_API_KEY = "gsk_..."
  python annotate_subset.py
  python annotate_subset.py --target-size 400
"""

import argparse
import json
import os
import sys
import logging
from collections import Counter
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Target subset proportions (match the original class distribution roughly)
TARGET_PROPORTIONS = {
    0: 0.09,   # negative
    1: 0.59,   # neutral
    2: 0.32,   # positive
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Stratified subset annotation: top up the smallest classes first."
    )
    p.add_argument("--target-size", type=int, default=380,
                   help="Target number of train+val samples to annotate (default: 380)")
    p.add_argument("--cache-path", type=str,
                   default="outputs/teacher_cache_groq.json",
                   help="Path to teacher annotation cache")
    p.add_argument("--audit-only", action="store_true",
                   help="Just print the audit, don't annotate")
    return p.parse_args()


def audit_cache(cache: dict, samples: list) -> dict:
    """
    Look up which samples have been annotated already (by matching sentence text).
    Return per-class counts.
    """
    by_class = {0: [], 1: [], 2: []}        # already annotated
    pending  = {0: [], 1: [], 2: []}        # not yet annotated

    for s in samples:
        if s.text in cache:
            by_class[s.label].append(s)
        else:
            pending[s.label].append(s)

    return {
        "by_class": by_class,
        "pending": pending,
        "annotated_total": sum(len(v) for v in by_class.values()),
        "pending_total": sum(len(v) for v in pending.values()),
    }


def select_to_annotate(audit: dict, target_size: int) -> list:
    """
    Decide which pending samples to annotate so that the final annotated set
    approximates TARGET_PROPORTIONS.
    """
    annotated = {c: len(audit["by_class"][c]) for c in [0, 1, 2]}

    # Target absolute counts
    targets = {c: int(target_size * p) for c, p in TARGET_PROPORTIONS.items()}
    needed  = {c: max(0, targets[c] - annotated[c]) for c in [0, 1, 2]}

    print(f"  Class targets at size {target_size}: {targets}")
    print(f"  Already annotated:                   {annotated}")
    print(f"  Need to annotate (this run):         {needed}")

    # Pick the needed count from each class's pending samples
    to_annotate = []
    for c in [0, 1, 2]:
        pending = audit["pending"][c]
        n_take = min(needed[c], len(pending))
        to_annotate.extend(pending[:n_take])

    return to_annotate


def main():
    args = parse_args()

    print("\n" + "█" * 60)
    print("  P6: STRATIFIED SUBSET ANNOTATION")
    print("█" * 60 + "\n")

    # ── 1. Load data ──────────────────────────────────────────────────────
    print("► Loading FinancialPhraseBank...")
    from src.data.loader import FinancialPhraseBankLoader
    loader = FinancialPhraseBankLoader(seed=42)
    split = loader.load()
    all_train_val = split.train + split.val
    print(f"  Total train+val samples: {len(all_train_val)}")
    print(f"  Test (untouched):        {len(split.test)}")

    # ── 2. Load cache ─────────────────────────────────────────────────────
    cache_path = Path(args.cache_path)
    cache = {}
    if cache_path.exists():
        with open(cache_path) as f:
            cache = json.load(f)
    print(f"\n► Cache: {cache_path}")
    print(f"  Cached annotations: {len(cache)}")

    # ── 3. Audit ──────────────────────────────────────────────────────────
    print(f"\n► Audit of existing annotations:")
    audit = audit_cache(cache, all_train_val)

    annotated_by_class = {c: len(v) for c, v in audit["by_class"].items()}
    pending_by_class = {c: len(v) for c, v in audit["pending"].items()}

    print(f"  Annotated:   "
          f"neg={annotated_by_class[0]:4d}  "
          f"neu={annotated_by_class[1]:4d}  "
          f"pos={annotated_by_class[2]:4d}  "
          f"total={audit['annotated_total']}")
    print(f"  Pending:     "
          f"neg={pending_by_class[0]:4d}  "
          f"neu={pending_by_class[1]:4d}  "
          f"pos={pending_by_class[2]:4d}  "
          f"total={audit['pending_total']}")

    # ── 4. Selection ──────────────────────────────────────────────────────
    print(f"\n► Selecting samples to annotate (target subset size = {args.target_size}):")
    to_annotate = select_to_annotate(audit, args.target_size)
    print(f"  Will annotate: {len(to_annotate)} new samples this session")

    if args.audit_only:
        print("\n► --audit-only flag set. Exiting without annotating.")
        return

    if not to_annotate:
        print("\n✓ Subset target already met. Nothing to do!")
        print("  Ready to train. Run main.py --teacher groq")
        return

    # ── 5. Annotate ───────────────────────────────────────────────────────
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        print("\n✗ GROQ_API_KEY environment variable not set.")
        print('  Run: $env:GROQ_API_KEY = "gsk_..."')
        sys.exit(1)

    from src.models.teacher import GroqTeacher
    teacher = GroqTeacher(api_key=groq_key, cache_path=str(cache_path))

    print(f"\n► Starting annotation. Estimated time: ~{len(to_annotate) * 2 / 60:.1f} min")
    print(f"  (Free-tier quota may interrupt; cache is saved progressively.)\n")

    teacher.annotate_batch(to_annotate, verbose=True)

    # ── 6. Final summary ──────────────────────────────────────────────────
    print(f"\n► Re-auditing after annotation:")
    # Reload cache (it may have been updated during annotation)
    with open(cache_path) as f:
        cache = json.load(f)
    audit = audit_cache(cache, all_train_val)

    annotated_by_class = {c: len(v) for c, v in audit["by_class"].items()}
    targets = {c: int(args.target_size * p) for c, p in TARGET_PROPORTIONS.items()}

    print(f"  Annotated now:   "
          f"neg={annotated_by_class[0]:4d}  "
          f"neu={annotated_by_class[1]:4d}  "
          f"pos={annotated_by_class[2]:4d}  "
          f"total={audit['annotated_total']}")
    print(f"  Target counts:   "
          f"neg={targets[0]:4d}  "
          f"neu={targets[1]:4d}  "
          f"pos={targets[2]:4d}  "
          f"total={args.target_size}")

    remaining_needed = sum(
        max(0, targets[c] - annotated_by_class[c]) for c in [0, 1, 2]
    )
    if remaining_needed == 0:
        print(f"\n  ✓ Stratified subset complete! Ready to train.")
    else:
        print(f"\n  ⚠ Still need {remaining_needed} more samples. Re-run tomorrow.")


if __name__ == "__main__":
    main()
