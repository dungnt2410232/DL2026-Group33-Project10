import argparse
import importlib.util
import json
import math
from pathlib import Path
import random


def load_script(filename, module_name):
    local = Path(__file__).resolve().parent / filename
    paths = [local] if local.is_file() else list(Path("/kaggle/input").rglob(filename))
    if len(paths) != 1:
        raise ValueError(f"Expected one {filename}; found {len(paths)}")
    spec = importlib.util.spec_from_file_location(module_name, paths[0])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def train(args):
    import torch
    import transformers
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm

    if not torch.cuda.is_available():
        raise RuntimeError("Enable GPU before fine-tuning")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    data_module = load_script("dataset.py", "vqa_dataset")
    model_module = load_script("model.py", "vqa_model")
    eval_module = load_script("evaluate.py", "vqa_evaluation")
    processor, model = model_module.load_model(args.model, "cuda")
    collator = data_module.VQACollator(processor, model.config.label2id)
    train_data = data_module.VQADataset(args.data_dir / "train.jsonl")
    dev_data = data_module.VQADataset(args.data_dir / "dev.jsonl")
    original_count = len(train_data)
    supported = set(collator.label2id)
    train_data.records = [
        r for r in train_data.records
        if any(data_module.answer_key(a) in supported for a in r["answers"])
    ]
    retained_count = len(train_data)
    if not retained_count:
        raise ValueError("No supported training labels")
    if args.limit_train:
        # Deterministic smoke-test sample, not the main experiment.
        train_data.records = random.Random(args.seed).sample(
            train_data.records, min(args.limit_train, len(train_data)))
    if args.limit_dev:
        dev_data.records = dev_data.records[:args.limit_dev]
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_data, batch_size=args.batch_size,
                              shuffle=True, generator=generator, num_workers=0,
                              collate_fn=collator)
    dev_loader = DataLoader(dev_data, batch_size=args.batch_size,
                            shuffle=False, num_workers=0, collate_fn=collator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                 weight_decay=0.01)
    total_steps = args.epochs * math.ceil(len(train_loader) / args.accumulation_steps)
    warmup_steps = int(total_steps * 0.1)

    def lr_multiplier(step):
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    scaler = torch.amp.GradScaler("cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update({"original_train_examples": original_count,
                   "excluded_unsupported_train_examples": original_count - retained_count,
                   "used_train_examples": len(train_data), "used_dev_examples": len(dev_data),
                   "partial_run": bool(args.limit_train or args.limit_dev),
                   "torch_version": torch.__version__,
                   "transformers_version": transformers.__version__,
                   "selection_metric": "dev consensus with lowercase/whitespace matching",
                   "model_revision": getattr(model.config, "_commit_hash", None)})
    (args.output_dir / "training_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"Train: {len(train_data)}; Dev: {len(dev_data)}; "
          f"excluded unsupported Train: {original_count-retained_count}", flush=True)

    history, best_score = [], -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum, example_count = 0.0, 0
        progress = tqdm(train_loader, desc=f"Train epoch {epoch}/{args.epochs}")
        for step, batch in enumerate(progress):
            inputs = {k: v.to("cuda") for k, v in batch["inputs"].items()}
            group_start = (step // args.accumulation_steps) * args.accumulation_steps
            group_size = min(args.accumulation_steps, len(train_loader) - group_start)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = model(**inputs).loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at epoch {epoch}, batch {step}")
            scaler.scale(loss / group_size).backward()
            size = len(batch["metadata"])
            loss_sum += loss.detach().item() * size
            example_count += size
            if (step + 1) % args.accumulation_steps == 0 or step + 1 == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= previous_scale:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            progress.set_postfix(loss=f"{loss_sum/example_count:.3f}")

        model.eval()
        dev_consensus, dev_exact, dev_count = 0.0, 0.0, 0
        with torch.inference_mode():
            for batch in tqdm(dev_loader, desc="Development evaluation"):
                inputs = {k: v.to("cuda") for k, v in batch["inputs"].items() if k != "labels"}
                # Match baseline evaluation precision.
                ids = model(**inputs).logits.argmax(dim=-1).cpu().tolist()
                for r, idx in zip(batch["metadata"], ids):
                    prediction = model.config.id2label[idx]
                    dev_consensus += eval_module.consensus_score(prediction, r["answers"])
                    dev_exact += float(eval_module.answer_key(prediction) ==
                                       eval_module.answer_key(r["answer"]))
                    dev_count += 1
        score = dev_consensus / dev_count
        row = {"epoch": epoch, "train_loss": loss_sum/example_count,
               "dev_exact_match_basic_pct": 100 * dev_exact/dev_count,
               "dev_consensus_basic_pct": 100 * score}
        history.append(row)
        print(json.dumps(row), flush=True)
        if score > best_score:
            best_score = score
            best_dir = args.output_dir / "best"
            model.save_pretrained(best_dir, safe_serialization=True)
            processor.save_pretrained(best_dir)
            (args.output_dir / "best_checkpoint.json").write_text(
                json.dumps(row, indent=2) + "\n", encoding="utf-8")
            print("Best checkpoint saved:", best_dir, flush=True)
        (args.output_dir / "training_history.json").write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8")
    print("Training complete. Test data was not used for selection.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="dandelin/vilt-b32-finetuned-vqa")
    parser.add_argument("--data-dir", type=Path, default=Path("/kaggle/working/vqa_processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/vilt_finetuned"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-dev", type=int, default=0)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.accumulation_steps) < 1:
        parser.error("epochs, batch-size, and accumulation-steps must be positive")
    if min(args.limit_train, args.limit_dev) < 0 or args.learning_rate <= 0:
        parser.error("limits must be nonnegative and learning-rate must be positive")
    train(args)
