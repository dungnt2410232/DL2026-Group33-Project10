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