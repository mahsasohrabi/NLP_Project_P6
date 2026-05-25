import re
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from collections import Counter

try:
    import torch
    from torch.utils.data import Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    Dataset = object  

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = 3



@dataclass
class FinancialSample:
   
    text: str
    label: int          # 0=negative, 1=neutral, 2=positive
    label_str: str      # readable label
    soft_labels: Optional[list] = None   # teacher probability distribution [neg, neu, pos]
    source: str = "FinancialPhraseBank"


@dataclass
class DataSplit:
    """Train / validation / test split."""
    train: list = field(default_factory=list)
    val: list = field(default_factory=list)
    test: list = field(default_factory=list)

    def stats(self) -> dict:
        def _count(samples):
            c = Counter(s.label_str for s in samples)
            return dict(c)
        return {
            "train": {"total": len(self.train), "distribution": _count(self.train)},
            "val":   {"total": len(self.val),   "distribution": _count(self.val)},
            "test":  {"total": len(self.test),  "distribution": _count(self.test)},
        }


# Loader

class FinancialPhraseBankLoader:

    HF_DATASET_NAME = "takala/financial_phrasebank"
    HF_CONFIG = "sentences_allagree"   

    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)

    #Public API 

    def load(self) -> DataSplit:
        raw_samples = self._fetch_from_hub()
        return self._stratified_split(raw_samples)

    def load_from_file(self, filepath: str) -> DataSplit:
        
        samples = []
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {filepath}")

        with open(path, encoding="latin-1") as f:
            for line in f:
                line = line.strip()
                if not line or "@" not in line:
                    continue
                parts = line.rsplit("@", 1)
                if len(parts) != 2:
                    continue
                text, label_str = parts[0].strip(), parts[1].strip().lower()
                if label_str not in LABEL2ID:
                    continue
                samples.append(FinancialSample(
                    text=self._clean(text),
                    label=LABEL2ID[label_str],
                    label_str=label_str,
                ))
        return self._stratified_split(samples)

    
    def _fetch_from_hub(self) -> list:
        
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError(
                "Install 'huggingface_hub' package: pip install huggingface_hub"
            )
        import zipfile

       
        zip_path = hf_hub_download(
            repo_id=self.HF_DATASET_NAME,
            filename="data/FinancialPhraseBank-v1.0.zip",
            repo_type="dataset",
        )

       
        config_to_filename = {
            "sentences_allagree":  "FinancialPhraseBank-v1.0/Sentences_AllAgree.txt",
            "sentences_75agree":   "FinancialPhraseBank-v1.0/Sentences_75Agree.txt",
            "sentences_66agree":   "FinancialPhraseBank-v1.0/Sentences_66Agree.txt",
            "sentences_50agree":   "FinancialPhraseBank-v1.0/Sentences_50Agree.txt",
        }
        if self.HF_CONFIG not in config_to_filename:
            raise ValueError(
                f"Unknown config '{self.HF_CONFIG}'. "
                f"Expected one of: {list(config_to_filename.keys())}"
            )

        target_file = config_to_filename[self.HF_CONFIG]
        samples = []

        with zipfile.ZipFile(zip_path) as z:
            # Original FinancialPhraseBank uses Latin-1 encoding
            with z.open(target_file) as f:
                for raw_line in f:
                    line = raw_line.decode("latin-1").strip()
                    if not line:
                        continue
                    if "@" not in line:
                        continue
                    sentence, label_str = line.rsplit("@", 1)
                    label_str = label_str.strip().lower()
                    if label_str not in LABEL2ID:
                        continue
                    label_int = LABEL2ID[label_str]
                    samples.append(FinancialSample(
                        text=self._clean(sentence),
                        label=label_int,
                        label_str=label_str,
                    ))
        return samples

    def _clean(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _stratified_split(
        self,
        samples: list,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
    ) -> DataSplit:
        
        # Group by label
        by_label: dict[int, list] = {0: [], 1: [], 2: []}
        for s in samples:
            by_label[s.label].append(s)

        train, val, test = [], [], []
        for label_id, group in by_label.items():
            random.shuffle(group)
            n = len(group)
            n_train = int(n * train_ratio)
            n_val   = int(n * val_ratio)
            train.extend(group[:n_train])
            val.extend(group[n_train:n_train + n_val])
            test.extend(group[n_train + n_val:])

        # Shuffle each split
        random.shuffle(train)
        random.shuffle(val)
        random.shuffle(test)

        return DataSplit(train=train, val=val, test=test)


# ── PyTorch Dataset wrapper ──────────────────────────────────────────────────

if HAS_TORCH:
    class FinancialSentimentDataset(Dataset):
        """
        PyTorch Dataset wrapping FinancialSample objects.
        Tokenizes on-the-fly using a provided tokenizer.
        Supports both hard labels (for baseline) and soft labels (for distillation).
        """

        def __init__(
            self,
            samples: list,
            tokenizer,
            max_length: int = 128,
            use_soft_labels: bool = False,
        ):
            self.samples = samples
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.use_soft_labels = use_soft_labels

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            sample = self.samples[idx]
            encoding = self.tokenizer(
                sample.text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            item = {
                "input_ids":      encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "labels":         torch.tensor(sample.label, dtype=torch.long),
            }
            if self.use_soft_labels and sample.soft_labels is not None:
                item["soft_labels"] = torch.tensor(
                    sample.soft_labels, dtype=torch.float
                )
            return item
