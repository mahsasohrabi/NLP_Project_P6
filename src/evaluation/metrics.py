import time
import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        confusion_matrix,
        classification_report,
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


@dataclass
class ModelMetrics:
    model_name: str
    mode: str                          

    # Performance
    accuracy: float = 0.0
    f1_macro: float = 0.0
    f1_negative: float = 0.0
    f1_neutral: float = 0.0
    f1_positive: float = 0.0

    # Efficiency
    n_parameters: int = 0
    model_size_mb: float = 0.0
    inference_ms_per_sample: float = 0.0

    # Derived
    f1_per_million_params: float = 0.0  

    # Detail
    confusion: list = field(default_factory=list)
    class_report: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def summary_line(self) -> str:
        return (
            f"[{self.mode:12s}] "
            f"acc={self.accuracy:.4f}  "
            f"f1={self.f1_macro:.4f}  "
            f"params={self.n_parameters/1e6:.1f}M  "
            f"speed={self.inference_ms_per_sample:.1f}ms/sample"
        )


@dataclass
class ComparisonReport:
    baseline: ModelMetrics
    distilled: ModelMetrics
    teacher_f1: Optional[float] = None   # upper bound reference

    def retention_ratio(self) -> float:
        if self.baseline.f1_macro == 0:
            return 0.0
        return self.distilled.f1_macro / self.baseline.f1_macro

    def f1_delta(self) -> float:
        return self.distilled.f1_macro - self.baseline.f1_macro

    def to_dict(self) -> dict:
        return {
            "baseline": self.baseline.to_dict(),
            "distilled": self.distilled.to_dict(),
            "teacher_f1_reference": self.teacher_f1,
            "f1_delta": round(self.f1_delta(), 4),
            "retention_ratio": round(self.retention_ratio(), 4),
        }

    def print_summary(self):
        print("\n" + "="*65)
        print("  EVALUATION SUMMARY")
        print("="*65)
        print(self.baseline.summary_line())
        print(self.distilled.summary_line())
        if self.teacher_f1:
            print(f"  Teacher F1 (upper bound): {self.teacher_f1:.4f}")
        print("-"*65)
        delta = self.f1_delta()
        sign = "+" if delta >= 0 else ""
        print(f"  F1 Δ (distilled - baseline):  {sign}{delta:.4f}")
        print(f"  Retention ratio:              {self.retention_ratio():.4f}")
        print("="*65)



class Evaluator:
   
    LABEL_NAMES = ["negative", "neutral", "positive"]

    def __init__(self, output_dir: str = "outputs/evaluation"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # API

    def evaluate_student(
        self,
        student,          # StudentModel instance
        test_samples: list,
        model_name: str,
        mode: str,
    ) -> ModelMetrics:
        
        texts  = [s.text for s in test_samples]
        labels = [s.label for s in test_samples]

        # Timed inference
        t0 = time.perf_counter()
        predictions = student.predict(texts)
        elapsed_ms  = (time.perf_counter() - t0) * 1000
        ms_per_sample = elapsed_ms / len(texts)

        pred_labels = [
            {"negative": 0, "neutral": 1, "positive": 2}[p["label"]]
            for p in predictions
        ]

        metrics = self._compute_metrics(labels, pred_labels)
        size_info = self._model_size(student)

        m = ModelMetrics(
            model_name=model_name,
            mode=mode,
            accuracy=metrics["accuracy"],
            f1_macro=metrics["f1_macro"],
            f1_negative=metrics["f1_per_class"][0],
            f1_neutral=metrics["f1_per_class"][1],
            f1_positive=metrics["f1_per_class"][2],
            n_parameters=size_info["n_parameters"],
            model_size_mb=size_info["size_mb"],
            inference_ms_per_sample=ms_per_sample,
            f1_per_million_params=metrics["f1_macro"] / (size_info["n_parameters"] / 1e6),
            confusion=metrics["confusion"],
            class_report=metrics["class_report"],
        )
        return m

    def evaluate_teacher_on_test(
        self,
        test_samples: list,
        teacher,
    ) -> float:
        
        true_labels = [s.label for s in test_samples]
        pred_labels = []

        for s in test_samples:
            probs = teacher.annotate(s.text)
            pred_labels.append(int(probs.index(max(probs))))

        return f1_score(true_labels, pred_labels, average="macro")

    def compare(
        self,
        baseline: ModelMetrics,
        distilled: ModelMetrics,
        teacher_f1: Optional[float] = None,
    ) -> ComparisonReport:
        
        report = ComparisonReport(
            baseline=baseline,
            distilled=distilled,
            teacher_f1=teacher_f1,
        )
        # Save JSON
        out_path = self.output_dir / "comparison_report.json"
        with open(out_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        logger.info(f"Comparison report saved to {out_path}")
        return report

    def save_metrics(self, metrics: ModelMetrics, filename: str):
        """Save individual model metrics to JSON."""
        out_path = self.output_dir / filename
        with open(out_path, "w") as f:
            json.dump(metrics.to_dict(), f, indent=2)


    def _compute_metrics(
        self,
        true_labels: list,
        pred_labels: list,
    ) -> dict:
        if not HAS_SKLEARN:
            raise ImportError("Install scikit-learn: pip install scikit-learn")

        acc        = accuracy_score(true_labels, pred_labels)
        f1_macro   = f1_score(true_labels, pred_labels, average="macro")
        f1_classes = f1_score(true_labels, pred_labels, average=None)
        cm         = confusion_matrix(true_labels, pred_labels).tolist()
        report     = classification_report(
            true_labels, pred_labels,
            target_names=self.LABEL_NAMES,
        )

        return {
            "accuracy":      acc,
            "f1_macro":      f1_macro,
            "f1_per_class":  list(f1_classes),
            "confusion":     cm,
            "class_report":  report,
        }

    def _model_size(self, student) -> dict:
        try:
            n_params = sum(p.numel() for p in student.model.parameters())
            size_mb = (n_params * 4) / (1024 ** 2)
            return {"n_parameters": n_params, "size_mb": round(size_mb, 1)}
        except Exception:
            return {"n_parameters": 0, "size_mb": 0.0}



def compute_teacher_agreement(
    samples: list,
    threshold: float = 0.6,
) -> dict:
   
    if not samples or samples[0].soft_labels is None:
        return {"error": "No soft labels found. Run teacher annotation first."}

    agreements = []      # does teacher agree with hard label?
    confidences = []
    high_conf_count = 0

    for s in samples:
        probs = s.soft_labels
        pred = probs.index(max(probs))
        agreements.append(int(pred == s.label))
        confidences.append(max(probs))
        if max(probs) >= threshold:
            high_conf_count += 1

    return {
        "total_samples": len(samples),
        "teacher_hard_agreement": round(sum(agreements) / len(agreements), 4),
        "mean_confidence": round(sum(confidences) / len(confidences), 4),
        "high_confidence_ratio": round(high_conf_count / len(samples), 4),
        "high_confidence_threshold": threshold,
    }
