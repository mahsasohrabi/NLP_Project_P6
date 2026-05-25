import logging
import math
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


@dataclass
class StudentConfig:
    
    # Model
    model_name: str = "distilbert-base-uncased"
    num_labels: int = 3
    max_length: int = 128

    # Distillation
    temperature: float = 4.0        
    alpha: float = 0.7             

    # Training
    learning_rate: float = 2e-5
    batch_size: int = 16
    num_epochs: int = 5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Output
    output_dir: str = "outputs/student"
    seed: int = 42

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "StudentConfig":
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


#Loss functions 

def distillation_loss(
    student_logits: "torch.Tensor",
    teacher_probs: "torch.Tensor",
    hard_labels: "torch.Tensor",
    temperature: float,
    alpha: float,
) -> "torch.Tensor":
    
    student_log_probs_T = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs_T = F.softmax(
        torch.log(teacher_probs.clamp(min=1e-8)) / temperature, dim=-1
    )
    kd_loss = F.kl_div(
        student_log_probs_T,
        teacher_probs_T,
        reduction="batchmean",
    ) * (temperature ** 2)

    ce_loss = F.cross_entropy(student_logits, hard_labels)

    return alpha * kd_loss + (1 - alpha) * ce_loss


# Student Model 

class StudentModel:
    
    def __init__(self, config: StudentConfig):
        if not HAS_TORCH or not HAS_TRANSFORMERS:
            raise ImportError(
                "Install torch and transformers:\n"
                "  pip install torch transformers"
            )
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps"  if torch.backends.mps.is_available() else
            "cpu"
        )
        self._set_seed(config.seed)

        # Load tokenizer and model
        logger.info(f"Loading {config.model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=config.num_labels,
        ).to(self.device)

        # Training history
        self.history: list[dict] = []

    #Training 

    def train(
        self,
        train_samples: list,
        val_samples: list,
        use_distillation: bool = True,
        verbose: bool = True,
    ) -> list[dict]:
        
        from src.data.loader import FinancialSentimentDataset

        # Build datasets
        train_ds = FinancialSentimentDataset(
            train_samples,
            self.tokenizer,
            max_length=self.config.max_length,
            use_soft_labels=use_distillation,
        )
        val_ds = FinancialSentimentDataset(
            val_samples,
            self.tokenizer,
            max_length=self.config.max_length,
            use_soft_labels=False,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.config.batch_size * 2,
            shuffle=False,
            num_workers=0,
        )

        # Optimizer & scheduler
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        total_steps = len(train_loader) * self.config.num_epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        mode = "Distillation" if use_distillation else "Baseline"
        if verbose:
            print(f"\n{'='*60}")
            print(f"  Training mode: {mode}")
            print(f"  Model: {self.config.model_name}")
            print(f"  Device: {self.device}")
            print(f"  Train samples: {len(train_samples)}")
            print(f"  Epochs: {self.config.num_epochs}")
            print(f"{'='*60}")

        best_val_acc = 0.0
        self.history = []

        for epoch in range(1, self.config.num_epochs + 1):
            train_loss = self._train_epoch(
                train_loader, optimizer, scheduler, use_distillation
            )
            val_metrics = self._evaluate(val_loader)

            record = {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_metrics["loss"], 4),
                "val_accuracy": round(val_metrics["accuracy"], 4),
                "val_f1_macro": round(val_metrics["f1_macro"], 4),
                "mode": mode,
            }
            self.history.append(record)

            if verbose:
                print(
                    f"Epoch {epoch}/{self.config.num_epochs}  "
                    f"train_loss={train_loss:.4f}  "
                    f"val_acc={val_metrics['accuracy']:.4f}  "
                    f"val_f1={val_metrics['f1_macro']:.4f}"
                )

            # Save best checkpoint
            if val_metrics["accuracy"] > best_val_acc:
                best_val_acc = val_metrics["accuracy"]
                self.save(Path(self.config.output_dir) / "best_checkpoint")

        if verbose:
            print(f"\n✓ Training complete. Best val accuracy: {best_val_acc:.4f}")

        return self.history

    # Inference 

    def predict(self, texts: list[str]) -> list[dict]:
        
        from src.data.loader import ID2LABEL
        self.model.eval()
        results = []

        for text in texts:
            enc = self.tokenizer(
                text,
                max_length=self.config.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                logits = self.model(
                    input_ids=enc["input_ids"].to(self.device),
                    attention_mask=enc["attention_mask"].to(self.device),
                ).logits
            probs = F.softmax(logits, dim=-1).squeeze().cpu().tolist()
            pred = int(torch.argmax(logits, dim=-1).item())
            results.append({
                "label": ID2LABEL[pred],
                "confidence": round(probs[pred], 4),
                "probs": {
                    "negative": round(probs[0], 4),
                    "neutral":  round(probs[1], 4),
                    "positive": round(probs[2], 4),
                },
            })
        return results


    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        self.config.save(path / "config.json")
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "StudentModel":
        config = StudentConfig.load(Path(path) / "config.json")
        student = cls(config)
        student.model = AutoModelForSequenceClassification.from_pretrained(path).to(
            student.device
        )
        student.tokenizer = AutoTokenizer.from_pretrained(path)
        return student


    def _train_epoch(
        self,
        loader: "DataLoader",
        optimizer,
        scheduler,
        use_distillation: bool,
    ) -> float:
        self.model.train()
        total_loss = 0.0

        for batch in loader:
            optimizer.zero_grad()

            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels         = batch["labels"].to(self.device)

            logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits

            if use_distillation and "soft_labels" in batch:
                soft_labels = batch["soft_labels"].to(self.device)
                loss = distillation_loss(
                    logits,
                    soft_labels,
                    labels,
                    temperature=self.config.temperature,
                    alpha=self.config.alpha,
                )
            else:
                loss = F.cross_entropy(logits, labels)

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    def _evaluate(self, loader: "DataLoader") -> dict:
        """Returns loss, accuracy, and macro-F1."""
        from sklearn.metrics import f1_score, accuracy_score

        self.model.eval()
        all_preds, all_labels = [], []
        total_loss = 0.0

        with torch.no_grad():
            for batch in loader:
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels         = batch["labels"].to(self.device)

                logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).logits
                loss = F.cross_entropy(logits, labels)
                total_loss += loss.item()

                preds = torch.argmax(logits, dim=-1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().tolist())

        return {
            "loss": total_loss / len(loader),
            "accuracy": accuracy_score(all_labels, all_preds),
            "f1_macro": f1_score(all_labels, all_preds, average="macro"),
        }

    def _set_seed(self, seed: int):
        import random, numpy as np
        random.seed(seed)
        np.random.seed(seed)
        if HAS_TORCH:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
