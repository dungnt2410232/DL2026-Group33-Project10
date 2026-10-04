from collections import Counter
import json
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset


def answer_key(answer):
    """Basic matching of already-normalized official answers, not VQA evaluation."""
    return str(answer).lower().strip()


class VQADataset(Dataset):
    def __init__(self, jsonl_path, image_dir=None):
        self.image_dir = Path(image_dir) if image_dir is not None else None
        with Path(jsonl_path).open(encoding="utf-8") as f:
            self.records = [json.loads(line) for line in f if line.strip()]
        if not self.records:
            raise ValueError(f"No records in {jsonl_path}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        path = Path(record["image_path"])
        # Override the old root when a Kaggle input location has changed.
        if self.image_dir is not None:
            path = self.image_dir / path.name
        with Image.open(path) as image:
            rgb_image = image.convert("RGB")
        return {"image": rgb_image, "record": record}


class VQACollator:
    def __init__(self, processor, label2id, max_length=40):
        self.processor = processor
        self.label2id = {answer_key(a): int(i) for a, i in label2id.items()}
        if len(self.label2id) != len(label2id):
            raise ValueError("Answer normalization creates duplicate vocabulary labels")
        if set(self.label2id.values()) != set(range(len(self.label2id))):
            raise ValueError("Answer IDs must be contiguous from zero")
        self.max_length = max_length

    def __call__(self, samples):
        records = [sample["record"] for sample in samples]
        inputs = self.processor(
            images=[sample["image"] for sample in samples],
            text=[record["question"] for record in records],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = torch.zeros((len(samples), len(self.label2id)), dtype=torch.float32)
        for row, record in enumerate(records):
            counts = Counter(answer_key(a) for a in record["answers"])
            for answer, count in counts.items():
                answer_id = self.label2id.get(answer)
                if answer_id is not None:
                    # Training target convention, not the official evaluation metric.
                    labels[row, answer_id] = min(count / 3.0, 1.0)
        inputs["labels"] = labels
        return {
            "inputs": inputs,
            "metadata": records,
            # Keep these examples in evaluation; training must handle them explicitly.
            "has_supported_answer": labels.sum(dim=1) > 0,
        }
