"""Prepare the available-image subset of official VQA v2 validation data."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re


CATEGORIES = ["counting", "color", "spatial", "reasoning", "object", "other"]


def classify_question(question):
    """Assign provisional categories using the notebook's rules."""
    q = question.lower().strip()
    if re.search(r"\bhow many\b|\bwhat number of\b", q):
        return "counting"
    if re.search(r"\bcolou?r\b", q):
        return "color"
    if q.startswith("why ") or re.search(r"\bpurpose of\b", q):
        return "reasoning"
    if re.search(
        r"^what\b.*\bcalled\b|^what is the name of\b|"
        r"^what items of clothing\b", q
    ):
        return "object"
    if re.search(
        r"\b(to the left of|to the right of|above|below|under|underneath|"
        r"behind|in front of|next to|beside|between|on top of)\b", q
    ):
        return "spatial"
    if re.search(
        r"^(what (animal|object|fruit|vehicle|food|sport|brand|instrument)\b|"
        r"what (kind|type) of (animal|object|fruit|vehicle|food|bird|instrument)\b|"
        r"what is (this|that)\s*\?)", q
    ):
        return "object"
    return "other"


def find_unique(root, name, directory=False):
    paths = [p for p in root.rglob(name)
             if (p.is_dir() if directory else p.is_file())]
    if len(paths) != 1:
        raise ValueError(f"Expected one {name}, found {len(paths)} under {root}")
    return paths[0]


def load_json(path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


def prepare(raw_dir, output_dir, seed):
    question_path = find_unique(raw_dir, "v2_OpenEnded_mscoco_val2014_questions.json")
    annotation_path = find_unique(raw_dir, "v2_mscoco_val2014_annotations.json")
    image_dir = find_unique(raw_dir, "val2014-resised", directory=True)
    questions = load_json(question_path)["questions"]
    annotations = load_json(annotation_path)["annotations"]
    answer_map = {a["question_id"]: a for a in annotations}
    if len(answer_map) != len(annotations):
        raise ValueError("Duplicate question IDs in annotations")
    if len({q["question_id"] for q in questions}) != len(questions):
        raise ValueError("Duplicate question IDs in questions")

    available_images = {p.name for p in image_dir.glob("*.jpg")}
    records, rejected = [], Counter()
    for q in questions:
        image_name = f"COCO_val2014_{q['image_id']:012d}.jpg"
        if image_name not in available_images:
            rejected["missing_image"] += 1
            continue
        a = answer_map.get(q["question_id"])
        if a is None:
            rejected["missing_annotation"] += 1
            continue
        if a["image_id"] != q["image_id"]:
            raise ValueError(f"Image ID mismatch for question {q['question_id']}")
        if not q["question"].strip() or not a["answers"]:
            rejected["empty_question_or_answers"] += 1
            continue
        records.append({
            "question_id": q["question_id"],
            "image_id": q["image_id"],
            "image_path": str((image_dir / image_name).resolve()),
            "question": q["question"],
            "answer": a["multiple_choice_answer"],
            "answers": [item["answer"] for item in a["answers"]],
            "answer_type": a["answer_type"],
            "category": classify_question(q["question"]),
        })
    if not records:
        raise ValueError("No usable records matched the available images")

    image_ids = sorted({r["image_id"] for r in records})
    random.Random(seed).shuffle(image_ids)
    n_train = int(len(image_ids) * 0.8)
    n_dev = int(len(image_ids) * 0.1)
    if min(n_train, n_dev, len(image_ids) - n_train - n_dev) == 0:
        raise ValueError("Not enough image groups for three nonempty splits")
    split_ids = {
        "train": image_ids[:n_train],
        "dev": image_ids[n_train:n_train + n_dev],
        "test": image_ids[n_train + n_dev:],
    }
    image_to_split = {i: split for split, ids in split_ids.items() for i in ids}
    splits = {name: [] for name in split_ids}
    for record in records:
        split = image_to_split[record["image_id"]]
        record["split"] = split
        splits[split].append(record)

    output_dir.mkdir(parents=True, exist_ok=True)
    stats = {}
    for name, rows in splits.items():
        with (output_dir / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        counts = Counter(r["category"] for r in rows)
        stats[name] = {
            "questions": len(rows), "images": len(split_ids[name]),
            "categories": {c: counts[c] for c in CATEGORIES},
            "answer_types": dict(Counter(r["answer_type"] for r in rows)),
        }
        print(f"{name}: {len(rows)} questions, {len(split_ids[name])} images")
        print("  Categories:", stats[name]["categories"])

    write_json(output_dir / "split_manifest.json", {
        "seed": seed, "ratios": [0.8, 0.1, 0.1],
        "rounding": "floor train/dev image counts; remainder to test",
        "image_ids": split_ids,
    })
    write_json(output_dir / "data_stats.json", {
        "source_questions": len(questions), "available_images": len(available_images),
        "matched_questions": len(records), "rejected": dict(rejected), "splits": stats,
    })
    write_json(output_dir / "dataset_metadata.json", {
        "dataset": "VQA v2.0 validation subset with available Kaggle images",
        "official_source": "https://visualqa.org/download.html",
        "image_source": "https://www.kaggle.com/datasets/henrychibueze/vqa-dataset",
        "source_sha256": {p.name: sha256(p) for p in [question_path, annotation_path]},
        "script_sha256": sha256(Path(__file__)),
        "image_directory": str(image_dir.resolve()),
        "category_method": "Provisional regex rules in classify_question; not human labels",
        "answer_processing": "Raw answers preserved; normalize later consistently with evaluator",
        "limitations": [
            "Image contents have not been checked for duplicates under different IDs",
            "Image decoding and original resize procedure have not been verified",
            "Project split separation does not establish no prior checkpoint exposure",
        ],
    })
    print("Prepared files saved to:", output_dir.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("/kaggle/working/vqa_processed"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(args.raw_dir, args.output_dir, args.seed)
