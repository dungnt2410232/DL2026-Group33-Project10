import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import ViltForQuestionAnswering, ViltProcessor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Checkpoint folder or Hugging Face model ID")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    if not args.image.is_file():
        parser.error(f"Image not found: {args.image}")
    if not args.question.strip():
        parser.error("Question must not be empty")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu")

    processor = ViltProcessor.from_pretrained(args.model)
    model = ViltForQuestionAnswering.from_pretrained(args.model)
    model.to(device).eval()

    with Image.open(args.image) as source:
        image = source.convert("RGB")
    inputs = processor(
        images=image, text=args.question, padding=True,
        truncation=True, max_length=40, return_tensors="pt"
    ).to(device)

    with torch.inference_mode():
        predicted_id = model(**inputs).logits.argmax(-1).item()

    print("Device:", device)
    print("Question:", args.question)
    print("Answer:", model.config.id2label[predicted_id])


if __name__ == "__main__":
    main()
