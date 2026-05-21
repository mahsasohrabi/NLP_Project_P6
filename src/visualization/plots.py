"""
Visualization module: training curves, confusion matrices,
soft label distributions, and model comparison charts.

All plots are saved as high-resolution PNG files suitable for the paper.
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.ticker as mticker
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ── Colour palette ────────────────────────────────────────────────────────────
PALETTE = {
    "baseline":  "#4361EE",
    "distilled": "#F72585",
    "teacher":   "#7209B7",
    "negative":  "#EF233C",
    "neutral":   "#8D99AE",
    "positive":  "#2EC4B6",
    "bg":        "#F8F9FA",
    "grid":      "#DEE2E6",
}

FIGSIZE_WIDE  = (12, 5)
FIGSIZE_SQ    = (7, 6)
DPI           = 150
FONT_TITLE    = 13
FONT_LABEL    = 11


def _style():
    """Apply consistent matplotlib style."""
    plt.rcParams.update({
        "font.family":      "DejaVu Sans",
        "axes.facecolor":   PALETTE["bg"],
        "figure.facecolor": "white",
        "axes.grid":        True,
        "grid.color":       PALETTE["grid"],
        "grid.linewidth":   0.8,
        "axes.spines.top":  False,
        "axes.spines.right":False,
    })


# ── 1. Training Curves ────────────────────────────────────────────────────────

def plot_training_curves(
    baseline_history: list[dict],
    distilled_history: list[dict],
    output_dir: str = "outputs/figures",
) -> str:
    """
    Plots train loss and val F1 side by side for baseline vs distilled.
    Returns path to saved figure.
    """
    if not HAS_MPL:
        logger.warning("matplotlib not installed — skipping plot")
        return ""

    _style()
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)
    fig.suptitle("Training Dynamics: Baseline vs Distilled Student", fontsize=FONT_TITLE + 1, fontweight="bold")

    epochs_b = [r["epoch"] for r in baseline_history]
    epochs_d = [r["epoch"] for r in distilled_history]

    # ── Left: Training loss ──
    ax = axes[0]
    ax.plot(epochs_b, [r["train_loss"] for r in baseline_history],
            color=PALETTE["baseline"], lw=2, label="Baseline (CE only)", marker="o", ms=5)
    ax.plot(epochs_d, [r["train_loss"] for r in distilled_history],
            color=PALETTE["distilled"], lw=2, label="Distilled (KD+CE)", marker="s", ms=5)
    ax.set_xlabel("Epoch", fontsize=FONT_LABEL)
    ax.set_ylabel("Training Loss", fontsize=FONT_LABEL)
    ax.set_title("Training Loss", fontsize=FONT_TITLE)
    ax.legend(fontsize=10)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # ── Right: Val F1 ──
    ax = axes[1]
    ax.plot(epochs_b, [r["val_f1_macro"] for r in baseline_history],
            color=PALETTE["baseline"], lw=2, label="Baseline", marker="o", ms=5)
    ax.plot(epochs_d, [r["val_f1_macro"] for r in distilled_history],
            color=PALETTE["distilled"], lw=2, label="Distilled", marker="s", ms=5)
    ax.set_xlabel("Epoch", fontsize=FONT_LABEL)
    ax.set_ylabel("Validation Macro-F1", fontsize=FONT_LABEL)
    ax.set_title("Validation Macro-F1", fontsize=FONT_TITLE)
    ax.legend(fontsize=10)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    path = _save(fig, "training_curves.png", output_dir)
    return path


# ── 2. Confusion Matrices ────────────────────────────────────────────────────

def plot_confusion_matrices(
    baseline_cm: list[list],
    distilled_cm: list[list],
    output_dir: str = "outputs/figures",
) -> str:
    """
    Side-by-side confusion matrices for baseline and distilled models.
    """
    if not HAS_MPL:
        return ""

    _style()
    labels = ["Negative", "Neutral", "Positive"]
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)
    fig.suptitle("Confusion Matrices on Test Set", fontsize=FONT_TITLE + 1, fontweight="bold")

    for ax, cm_data, title, color in zip(
        axes,
        [baseline_cm, distilled_cm],
        ["Baseline (CE only)", "Distilled (KD+CE)"],
        [PALETTE["baseline"], PALETTE["distilled"]],
    ):
        cm = np.array(cm_data)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel("Predicted", fontsize=FONT_LABEL)
        ax.set_ylabel("True", fontsize=FONT_LABEL)
        ax.set_title(title, fontsize=FONT_TITLE, color=color, fontweight="bold")

        for i in range(3):
            for j in range(3):
                val = cm[i, j]
                pct = cm_norm[i, j]
                text_color = "white" if pct > 0.55 else "black"
                ax.text(j, i, f"{val}\n({pct:.0%})",
                        ha="center", va="center", fontsize=9, color=text_color)

    plt.tight_layout()
    path = _save(fig, "confusion_matrices.png", output_dir)
    return path


# ── 3. Model Comparison Bar Chart ────────────────────────────────────────────

def plot_model_comparison(
    baseline_metrics,
    distilled_metrics,
    teacher_f1: Optional[float] = None,
    output_dir: str = "outputs/figures",
) -> str:
    """
    Bar chart comparing key metrics across model configurations.
    """
    if not HAS_MPL:
        return ""

    _style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Baseline vs Distilled Student — Test Set Comparison",
                 fontsize=FONT_TITLE + 1, fontweight="bold")

    models    = ["Baseline", "Distilled"]
    colors    = [PALETTE["baseline"], PALETTE["distilled"]]
    x         = np.arange(len(models))
    bar_width = 0.5

    # ── F1 Macro ──
    ax = axes[0]
    vals = [baseline_metrics.f1_macro, distilled_metrics.f1_macro]
    bars = ax.bar(x, vals, width=bar_width, color=colors, alpha=0.85, edgecolor="white", lw=1.5)
    if teacher_f1:
        ax.axhline(teacher_f1, color=PALETTE["teacher"], ls="--", lw=1.5,
                   label=f"Teacher F1 = {teacher_f1:.3f}")
        ax.legend(fontsize=9)
    ax.set_title("Macro-F1", fontsize=FONT_TITLE)
    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_ylim(0, 1)
    _label_bars(ax, bars)

    # ── Per-class F1 ──
    ax = axes[1]
    classes = ["Negative", "Neutral", "Positive"]
    class_colors = [PALETTE["negative"], PALETTE["neutral"], PALETTE["positive"]]
    base_vals = [baseline_metrics.f1_negative, baseline_metrics.f1_neutral, baseline_metrics.f1_positive]
    dist_vals = [distilled_metrics.f1_negative, distilled_metrics.f1_neutral, distilled_metrics.f1_positive]
    xc = np.arange(len(classes))
    bw = 0.35
    bars1 = ax.bar(xc - bw/2, base_vals, width=bw, color=PALETTE["baseline"], alpha=0.8, label="Baseline")
    bars2 = ax.bar(xc + bw/2, dist_vals, width=bw, color=PALETTE["distilled"], alpha=0.8, label="Distilled")
    ax.set_title("Per-class F1", fontsize=FONT_TITLE)
    ax.set_xticks(xc); ax.set_xticklabels(classes)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)

    # ── Inference Speed ──
    ax = axes[2]
    speeds = [baseline_metrics.inference_ms_per_sample, distilled_metrics.inference_ms_per_sample]
    bars3 = ax.bar(x, speeds, width=bar_width, color=colors, alpha=0.85, edgecolor="white", lw=1.5)
    ax.set_title("Inference Speed (ms/sample)", fontsize=FONT_TITLE)
    ax.set_xticks(x); ax.set_xticklabels(models)
    _label_bars(ax, bars3, fmt="{:.1f}")

    plt.tight_layout()
    path = _save(fig, "model_comparison.png", output_dir)
    return path


# ── 4. Soft Label Distribution ───────────────────────────────────────────────

def plot_soft_label_distribution(
    samples: list,
    n_examples: int = 200,
    output_dir: str = "outputs/figures",
) -> str:
    """
    Visualizes the distribution of teacher soft labels.
    Shows how "soft" the teacher's confidence is vs hard labels.
    """
    if not HAS_MPL:
        return ""

    annotated = [s for s in samples if s.soft_labels is not None]
    if not annotated:
        logger.warning("No annotated samples found for soft label plot.")
        return ""

    _style()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("Teacher Soft Label Distributions by True Class",
                 fontsize=FONT_TITLE + 1, fontweight="bold")

    label_names = ["Negative", "Neutral", "Positive"]
    label_colors = [PALETTE["negative"], PALETTE["neutral"], PALETTE["positive"]]

    for true_class in range(3):
        class_samples = [s for s in annotated if s.label == true_class][:n_examples]
        if not class_samples:
            continue

        probs = np.array([s.soft_labels for s in class_samples])   # [N, 3]
        ax = axes[true_class]

        for i, (name, color) in enumerate(zip(label_names, label_colors)):
            ax.hist(probs[:, i], bins=20, alpha=0.65, color=color,
                    label=name, edgecolor="white", lw=0.5)

        ax.set_title(f"True label: {label_names[true_class]}",
                     fontsize=FONT_TITLE, fontweight="bold")
        ax.set_xlabel("Probability assigned by teacher", fontsize=FONT_LABEL)
        ax.set_ylabel("Count", fontsize=FONT_LABEL)
        ax.legend(fontsize=9)
        ax.set_xlim(0, 1)

    plt.tight_layout()
    path = _save(fig, "soft_label_distribution.png", output_dir)
    return path


# ── 5. Efficiency Frontier ───────────────────────────────────────────────────

def plot_efficiency_scatter(
    model_sizes_m: list[float],
    f1_scores: list[float],
    model_labels: list[str],
    highlight: Optional[str] = None,
    output_dir: str = "outputs/figures",
) -> str:
    """
    Scatter plot of model size (M params) vs F1 — the classic
    efficiency frontier plot.
    """
    if not HAS_MPL:
        return ""

    _style()
    fig, ax = plt.subplots(figsize=FIGSIZE_SQ)

    color_map = {
        "Baseline":  PALETTE["baseline"],
        "Distilled": PALETTE["distilled"],
        "Teacher":   PALETTE["teacher"],
    }

    for size, f1, label in zip(model_sizes_m, f1_scores, model_labels):
        color = color_map.get(label, "#888")
        ax.scatter(size, f1, s=200, color=color, zorder=5,
                   edgecolors="white", lw=1.5, label=label)
        ax.annotate(label, (size, f1), textcoords="offset points",
                    xytext=(10, 5), fontsize=10, color=color, fontweight="bold")

    ax.set_xlabel("Model Size (Million Parameters)", fontsize=FONT_LABEL)
    ax.set_ylabel("Macro-F1 on Test Set", fontsize=FONT_LABEL)
    ax.set_title("Performance vs Model Size (Efficiency Frontier)",
                 fontsize=FONT_TITLE, fontweight="bold")
    ax.set_ylim(0, 1)

    plt.tight_layout()
    path = _save(fig, "efficiency_frontier.png", output_dir)
    return path


# ── Helpers ──────────────────────────────────────────────────────────────────

def _save(fig, filename: str, output_dir: str) -> str:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    full_path = str(path / filename)
    fig.savefig(full_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved figure: {full_path}")
    return full_path


def _label_bars(ax, bars, fmt: str = "{:.3f}"):
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.01,
            fmt.format(h),
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )
