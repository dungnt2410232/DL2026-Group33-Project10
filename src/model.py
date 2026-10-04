import torch
from transformers import ViltForQuestionAnswering, ViltProcessor


DEFAULT_CHECKPOINT = "dandelin/vilt-b32-finetuned-vqa"


def load_model(checkpoint=DEFAULT_CHECKPOINT, device=None):
    """Return (processor, model). Call model.train() explicitly for training."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = ViltProcessor.from_pretrained(checkpoint)
    model = ViltForQuestionAnswering.from_pretrained(checkpoint)
    model.to(device)
    model.eval()
    return processor, model
