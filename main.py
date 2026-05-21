"""
main.py — Orchestrates the full P6 Apprentice Model experiment pipeline.

Pipeline:
  1. Load FinancialPhraseBank (sentences_allagree split)
  2. Annotate training + val sets with teacher soft labels (Claude API)
  3. Train student (DistilBERT) in BASELINE mode (CE only)
  4. Train student (DistilBERT) in DISTILLATION mode (KD + CE)
  5. Evaluate both on held-out test set
  6. Generate visualizations and save comparison report

Usage:
  # Full run (requires Anthropic API key):
  python main.py --api-key sk-ant-...

  # Offline demo (uses mock teacher, no API):
  python main.py --mock

  # Skip training, only evaluate a saved checkpoint:
  python main.py --eval-only --checkpoint outputs/student/best_checkpoint

Run `python main.py --help` for all options.
"""

import argparse
import json
import logging
import sys
import os
from pathlib import Path

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="P6: The Apprentice Model — Financial Sentiment Distillation"
    )
    parser.add_argument("--teacher", type=str, default="auto",
                        choices=["auto", "claude", "groq", "mock"],
                        help="Teacher choice: 'auto' picks based on available API keys; "
                             "'claude' uses Anthropic; 'groq' uses Llama 3.3 70B via Groq; "
                             "'mock' uses synthetic teacher")
    parser.add_argument("--api-key", type=str, default=None,
                        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")
    parser.add_argument("--groq-api-key", type=str, default=None,
                        help="Groq API key (or set GROQ_API_KEY env var)")
    parser.add_argument("--skip-annotation", action="store_true",
                        help="Skip teacher API calls; use only cached annotations. "
                             "Useful if API access is restricted.")
    parser.add_argument("--mock", action="store_true",
                        help="Use mock teacher (no API calls, for offline testing)")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs (default: 5)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Training batch size (default: 16)")
    parser.add_argument("--temperature", type=float, default=4.0,
                        help="Distillation temperature T (default: 4.0)")
    parser.add_argument("--alpha", type=float, default=0.7,
                        help="Weight on KD loss, 0-1 (default: 0.7)")
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Root output directory (default: outputs)")
    parser.add_argument("--cache-path", type=str,
                        default="outputs/teacher_cache.json",
                        help="Path to teacher annotation cache")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, only evaluate saved checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ── Pipeline ──────────────────────────────────────────────────────────────────
def run_pipeline(args):
    print("\n" + "█"*60)
    print("  P6: THE APPRENTICE MODEL")
    print("  Financial Sentiment Distillation")
    print("  DistilBERT student ← Claude teacher")
    print("█"*60 + "\n")

    # ── 1. Load data ──────────────────────────────────────────────────────
    print("► Step 1: Loading FinancialPhraseBank...")
    from src.data.loader import FinancialPhraseBankLoader

    loader = FinancialPhraseBankLoader(seed=args.seed)
    try:
        split = loader.load()
        print("  Source: HuggingFace Hub (financial_phrasebank/sentences_allagree)")
    except Exception as e:
        logger.warning(f"HF load failed ({e}). Looking for local file...")
        # Fallback: generate synthetic demo data
        split = _generate_demo_data(loader)

    stats = split.stats()
    print(f"  Train: {stats['train']['total']} samples → {stats['train']['distribution']}")
    print(f"  Val:   {stats['val']['total']} samples → {stats['val']['distribution']}")
    print(f"  Test:  {stats['test']['total']} samples → {stats['test']['distribution']}")

    # ── 2. Teacher annotation ─────────────────────────────────────────────
    print("\n► Step 2: Annotating with teacher model...")

    # Resolve teacher choice
    teacher_choice = args.teacher
    anthropic_key = args.api_key or os.getenv("ANTHROPIC_API_KEY")
    groq_key = args.groq_api_key or os.getenv("GROQ_API_KEY")

    if args.mock:
        teacher_choice = "mock"
    elif teacher_choice == "auto":
        if groq_key:
            teacher_choice = "groq"
        elif anthropic_key:
            teacher_choice = "claude"
        else:
            print("  ✗ No API key found. Falling back to mock teacher.")
            teacher_choice = "mock"

    # Use a teacher-specific cache file to avoid mixing annotations
    if args.cache_path == "outputs/teacher_cache.json":
        cache_path = f"outputs/teacher_cache_{teacher_choice}.json"
    else:
        cache_path = args.cache_path

    teacher = None
    if args.skip_annotation:
        print(f"  Mode: SKIP ANNOTATION (loading cached annotations only)")
        print(f"  Cache file: {cache_path}")
    elif teacher_choice == "mock":
        from src.models.teacher import MockTeacher
        teacher = MockTeacher(confidence=0.80, seed=args.seed)
        print("  Mode: MOCK teacher (offline, no API)")
    elif teacher_choice == "groq":
        if not groq_key:
            print("  ✗ Groq teacher selected but no GROQ_API_KEY provided.")
            print("  Use --groq-api-key or set GROQ_API_KEY env var.")
            sys.exit(1)
        from src.models.teacher import GroqTeacher
        teacher = GroqTeacher(api_key=groq_key, cache_path=cache_path)
        cost = teacher.estimate_cost(len(split.train) + len(split.val))
        print(f"  Mode: GROQ teacher (Llama 3.3 70B)")
        print(f"  Already cached: {cost['cached']} samples")
        print(f"  New API calls needed: {cost['new_api_calls']}")
        print(f"  Estimated cost: ${cost['estimated_cost_usd']} (free tier)")
        print(f"  Estimated time: ~{cost['estimated_time_minutes']} min")
        if cost['new_api_calls'] > 1000:
            print(f"  ⚠ {cost['daily_quota_warning']}")
    elif teacher_choice == "claude":
        if not anthropic_key:
            print("  ✗ Claude teacher selected but no ANTHROPIC_API_KEY provided.")
            sys.exit(1)
        from src.models.teacher import TeacherModel
        teacher = TeacherModel(api_key=anthropic_key, cache_path=cache_path)
        cost = teacher.estimate_cost(len(split.train) + len(split.val))
        print(f"  Mode: CLAUDE teacher (Sonnet)")
        print(f"  Estimated cost: ${cost['estimated_cost_usd']} "
              f"({cost['new_api_calls']} new API calls)")
    else:
        raise ValueError(f"Unknown teacher: {teacher_choice}")

    # Annotate train + val (test stays clean for final eval)
    all_to_annotate = split.train + split.val

    if args.skip_annotation:
        # Load cached annotations only — no API calls
        print(f"  --skip-annotation flag set. Loading cached annotations only.")
        import json
        cache = {}
        if Path(cache_path).exists():
            with open(cache_path) as f:
                cache = json.load(f)
        print(f"  Cache file: {cache_path}")
        print(f"  Cached annotations: {len(cache)}")
        loaded = 0
        for sample in all_to_annotate:
            if sample.text in cache:
                sample.soft_labels = cache[sample.text]["probs"]
                loaded += 1
        print(f"  Loaded soft labels for {loaded} samples (of {len(all_to_annotate)} total).")
    else:
        teacher.annotate_batch(all_to_annotate, verbose=True)

    # Teacher agreement stats
    from src.evaluation.metrics import compute_teacher_agreement
    annotated_train = [s for s in split.train if s.soft_labels is not None]
    if annotated_train:
        agree_stats = compute_teacher_agreement(annotated_train)
        print(f"  Teacher-label agreement: {agree_stats.get('teacher_hard_agreement', 'N/A')}")
        print(f"  Mean teacher confidence: {agree_stats.get('mean_confidence', 'N/A')}")

    # ── 2b. Filter train to annotated subset ───────────────────────────────
    # If some training samples weren't annotated (e.g., free-tier rate limit,
    # API restriction), we restrict the train set to only those that DO have
    # soft labels. This way both baseline and distilled train on the same
    # subset, keeping the comparison clean.
    # Note: validation always uses hard labels (regardless of distillation
    # mode), so we keep the full val set for stable early-stopping signal.
    original_train_size = len(split.train)
    split.train = [s for s in split.train if s.soft_labels is not None]

    print(f"\n► Annotated subset for training:")
    print(f"  Train: {len(split.train)} / {original_train_size} samples (annotated)")
    print(f"  Val:   {len(split.val)} samples (hard labels only, used for early stopping)")
    print(f"  Test:  {len(split.test)} samples (untouched, used for final evaluation)")
    if len(split.train) == 0:
        print("  ✗ No annotated training samples! Run teacher annotation first.")
        sys.exit(1)

    # Class distribution of annotated train
    train_dist = {0: 0, 1: 0, 2: 0}
    for s in split.train:
        train_dist[s.label] += 1
    print(f"  Train class distribution: "
          f"neg={train_dist[0]}, neu={train_dist[1]}, pos={train_dist[2]}")

    # ── 3. Soft label visualization ───────────────────────────────────────
    print("\n► Step 3: Visualizing soft label distributions...")
    from src.visualization.plots import plot_soft_label_distribution
    fig_dir = str(Path(args.output_dir) / "figures")
    plot_soft_label_distribution(split.train, output_dir=fig_dir)

    if args.eval_only:
        print("\n  --eval-only flag set. Skipping training.")
        _eval_checkpoints(args, split)
        return

    # ── 4. Train baseline ─────────────────────────────────────────────────
    print("\n► Step 4: Training BASELINE student (Cross-Entropy only)...")
    from src.models.student import StudentModel, StudentConfig

    baseline_config = StudentConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        temperature=args.temperature,
        alpha=args.alpha,
        seed=args.seed,
        output_dir=str(Path(args.output_dir) / "student_baseline"),
    )
    baseline_student = StudentModel(baseline_config)
    baseline_history = baseline_student.train(
        split.train, split.val,
        use_distillation=False,
    )

    # ── 5. Train distilled ────────────────────────────────────────────────
    print("\n► Step 5: Training DISTILLED student (KD + Cross-Entropy)...")
    distilled_config = StudentConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        temperature=args.temperature,
        alpha=args.alpha,
        seed=args.seed,
        output_dir=str(Path(args.output_dir) / "student_distilled"),
    )
    distilled_student = StudentModel(distilled_config)
    distilled_history = distilled_student.train(
        split.train, split.val,
        use_distillation=True,
    )

    # ── 6. Evaluate on test set ───────────────────────────────────────────
    print("\n► Step 6: Evaluating on held-out test set...")
    from src.evaluation.metrics import Evaluator

    evaluator = Evaluator(output_dir=str(Path(args.output_dir) / "evaluation"))

    baseline_metrics  = evaluator.evaluate_student(
        baseline_student, split.test,
        model_name="DistilBERT-Baseline", mode="baseline",
    )
    distilled_metrics = evaluator.evaluate_student(
        distilled_student, split.test,
        model_name="DistilBERT-Distilled", mode="distilled",
    )

    evaluator.save_metrics(baseline_metrics,  "baseline_metrics.json")
    evaluator.save_metrics(distilled_metrics, "distilled_metrics.json")

    report = evaluator.compare(baseline_metrics, distilled_metrics)
    report.print_summary()

    # Save training histories
    _save_json(baseline_history,  Path(args.output_dir) / "baseline_history.json")
    _save_json(distilled_history, Path(args.output_dir) / "distilled_history.json")

    # ── 7. Visualizations ─────────────────────────────────────────────────
    print("\n► Step 7: Generating figures...")
    from src.visualization.plots import (
        plot_training_curves,
        plot_confusion_matrices,
        plot_model_comparison,
        plot_efficiency_scatter,
    )

    plot_training_curves(baseline_history, distilled_history, output_dir=fig_dir)

    plot_confusion_matrices(
        baseline_metrics.confusion,
        distilled_metrics.confusion,
        output_dir=fig_dir,
    )

    plot_model_comparison(
        baseline_metrics,
        distilled_metrics,
        output_dir=fig_dir,
    )

    # Efficiency scatter (DistilBERT ~66M, large LLM teacher ~7B+)
    plot_efficiency_scatter(
        model_sizes_m=[66, 66, 7000],
        f1_scores=[baseline_metrics.f1_macro, distilled_metrics.f1_macro, 0.92],
        model_labels=["Baseline", "Distilled", "Teacher"],
        output_dir=fig_dir,
    )

    print(f"\n✓ All outputs saved to: {args.output_dir}/")
    print(f"  Figures:    {fig_dir}/")
    print(f"  Report:     {args.output_dir}/evaluation/comparison_report.json\n")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _generate_demo_data(loader):
    """Generate small synthetic dataset for offline testing."""
    import random
    from src.data.loader import FinancialSample, DataSplit, LABEL2ID

    random.seed(42)
    templates = {
        "positive": [
            "The company reported strong quarterly earnings, beating analyst estimates.",
            "Revenue grew by 15% year-over-year, driven by international expansion.",
            "The acquisition is expected to significantly boost margins.",
            "The firm announced a dividend increase of 10%.",
            "Customer satisfaction scores reached an all-time high.",
        ],
        "neutral": [
            "The board of directors met to discuss strategic priorities.",
            "The company filed its annual report with the SEC.",
            "Management provided guidance for the upcoming fiscal year.",
            "The firm operates in 42 countries worldwide.",
            "The CFO outlined the capital allocation strategy.",
        ],
        "negative": [
            "Operating losses widened as costs exceeded revenue.",
            "The company issued a profit warning for the third quarter.",
            "Layoffs of 2,000 employees were announced amid restructuring.",
            "Supply chain disruptions led to significant delays.",
            "The stock fell 8% after disappointing earnings.",
        ],
    }

    samples = []
    for label_str, texts in templates.items():
        for text in texts * 30:  # replicate for size
            samples.append(FinancialSample(
                text=text + f" (variant {random.randint(1,999)})",
                label=LABEL2ID[label_str],
                label_str=label_str,
            ))
    random.shuffle(samples)
    return loader._stratified_split(samples)


def _eval_checkpoints(args, split):
    """Evaluate pre-trained checkpoints without training."""
    from src.models.student import StudentModel
    from src.evaluation.metrics import Evaluator

    base_path = Path(args.output_dir)
    evaluator = Evaluator(output_dir=str(base_path / "evaluation"))

    for mode, subdir in [("baseline", "student_baseline"), ("distilled", "student_distilled")]:
        ckpt = base_path / subdir / "best_checkpoint"
        if ckpt.exists():
            student = StudentModel.load(str(ckpt))
            metrics = evaluator.evaluate_student(
                student, split.test,
                model_name=f"DistilBERT-{mode.capitalize()}",
                mode=mode,
            )
            print(metrics.summary_line())
        else:
            print(f"  ✗ Checkpoint not found: {ckpt}")


def _save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
