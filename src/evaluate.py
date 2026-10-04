import argparse
from collections import defaultdict
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import time


GROUPS = ["counting", "color", "spatial", "reasoning", "object", "other"]


def answer_key(text):
    return " ".join(str(text).lower().split())


def consensus_score(prediction, answers):
    if len(answers) != 10:
        raise ValueError("Expected all 10 human answers for consensus scoring")
    prediction = answer_key(prediction)
    matches = sum(answer_key(a) == prediction for a in answers)
    return (
        matches * min(max(matches - 1, 0) / 3.0, 1.0)
        + (10 - matches) * min(matches / 3.0, 1.0)
    ) / 10.0


def find_script(name):
    local = Path(__file__).resolve().parent / name
    if local.is_file():
        return local
    paths = list(Path("/kaggle/input").rglob(name))
    if len(paths) != 1:
        raise ValueError(f"Expected one {name}; found {len(paths)}")
    return paths[0]


def load_script(name, module_name):
    spec = importlib.util.spec_from_file_location(module_name, find_script(name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate(args):
    import torch
    import transformers
    from torch.utils.data import DataLoader, Subset
    from tqdm.auto import tqdm

    dataset_module = load_script("dataset.py", "vqa_dataset")
    model_module = load_script("model.py", "vqa_model")
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    processor, model = model_module.load_model(args.model, args.device)
    device = next(model.parameters()).device
    dataset = dataset_module.VQADataset(args.data_file, args.image_dir)
    full_size = len(dataset)
    if args.limit:
        dataset = Subset(dataset, range(min(args.limit, full_size)))
    collator = dataset_module.VQACollator(processor, model.config.label2id)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, collate_fn=collator)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats = defaultdict(lambda: {"count": 0, "exact": 0.0, "consensus": 0.0,
                                 "unsupported": 0})
    predictions = []
    started = time.perf_counter()
    print(f"Device: {device}; evaluation examples: {len(dataset)}", flush=True)

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Evaluation"):
            inputs = {name: value.to(device)
                      for name, value in batch["inputs"].items() if name != "labels"}
            ids = model(**inputs).logits.argmax(dim=-1).cpu().tolist()
            supported = batch["has_supported_answer"].tolist()
            for record, answer_id, has_answer in zip(batch["metadata"], ids, supported):
                prediction = model.config.id2label[answer_id]
                exact = float(answer_key(prediction) == answer_key(record["answer"]))
                consensus = consensus_score(prediction, record["answers"])
                for group in ["overall", record["category"]]:
                    stats[group]["count"] += 1
                    stats[group]["exact"] += exact
                    stats[group]["consensus"] += consensus
                    stats[group]["unsupported"] += int(not has_answer)
                predictions.append({
                    "question_id": record["question_id"], "image_id": record["image_id"],
                    "question": record["question"], "category": record["category"],
                    "reference_answer": record["answer"], "answers": record["answers"],
                    "prediction": prediction, "exact_match_basic": exact,
                    "consensus_basic": consensus,
                })

    rows = []
    for group in ["overall"] + GROUPS:
        values = stats[group]
        count = values["count"]
        rows.append({
            "category": group, "count": count,
            "exact_match_basic_pct": 100 * values["exact"] / count if count else None,
            "consensus_basic_pct": 100 * values["consensus"] / count if count else None,
            "unsupported_examples": values["unsupported"],
        })
        if count:
            print(f"{group}: n={count}, exact={rows[-1]['exact_match_basic_pct']:.2f}%, "
                  f"consensus={rows[-1]['consensus_basic_pct']:.2f}%", flush=True)
        else:
            print(f"{group}: n=0, scores=N/A", flush=True)

    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for prediction in predictions:
            f.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    report = {
        "model": args.model, "model_revision": getattr(model.config, "_commit_hash", None),
        "data_file": str(args.data_file),
        "data_sha256": hashlib.sha256(args.data_file.read_bytes()).hexdigest(),
        "device": str(device), "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "batch_size": args.batch_size, "max_question_length": 40,
        "evaluated_examples": len(predictions), "available_examples": full_size,
        "partial_run": len(predictions) != full_size,
        "elapsed_seconds": time.perf_counter() - started,
        "normalization": "lowercase and collapse whitespace only",
        "official_vqa_evaluation": False,
        "metrics": rows,
        "macro_five_categories": {
            key: sum(r[key] for r in rows[1:6] if r["count"]) /
                 sum(bool(r["count"]) for r in rows[1:6])
                 if any(r["count"] for r in rows[1:6]) else None
            for key in ["exact_match_basic_pct", "consensus_basic_pct"]
        },
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "vqa_predictions.json").write_text(
        json.dumps([{"question_id": p["question_id"], "answer": p["prediction"]}
                    for p in predictions], ensure_ascii=False), encoding="utf-8")
    print("Results saved to:", args.output_dir.resolve(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="dandelin/vilt-b32-finetuned-vqa")
    parser.add_argument("--data-file", type=Path,
                        default=Path("/kaggle/working/vqa_processed/test.jsonl"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("/kaggle/working/results/baseline"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0,
                        help="Optional smoke-test limit; zero evaluates every record")
    args = parser.parse_args()
    if args.batch_size < 1 or args.limit < 0:
        parser.error("batch-size must be positive; limit must be nonnegative")
    evaluate(args)
