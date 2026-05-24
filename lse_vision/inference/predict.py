from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from lse_vision.config.settings import AppConfig, load_config
from lse_vision.preprocessing.transforms import build_eval_transforms, build_eval_transforms_imagenet
from lse_vision.training.common import build_model_from_checkpoint
from lse_vision.utils.checkpointing import load_checkpoint
from lse_vision.utils.runtime import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict a sign class from a single image")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    return parser.parse_args()


def run_predict(config: AppConfig, image_path: Path, checkpoint_path: Path) -> int:
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    class_names = checkpoint["classes"]

    model = build_model_from_checkpoint(checkpoint, config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    arch = checkpoint.get("architecture", "cnn")
    if arch == "mobilenetv2":
        transform = build_eval_transforms_imagenet(config.data.image_size)
    else:
        transform = build_eval_transforms(config.data.image_size)

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)

    device = torch.device(resolve_device())
    model = model.to(device)
    tensor = tensor.to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)
        k_value = min(config.inference.top_k, len(class_names))
        top_values, top_indices = torch.topk(probabilities, k=k_value, dim=1)

    result = {
        "image": str(image_path.resolve()),
        "top_predictions": [
            {
                "class_name": class_names[index],
                "confidence": float(score),
            }
            for score, index in zip(top_values[0].cpu().tolist(), top_indices[0].cpu().tolist())
        ],
    }
    print(json.dumps(result, indent=2))
    return 0


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    checkpoint_path = args.checkpoint or (config.paths.checkpoints_dir / "best_static_sign_cnn.pt")
    return run_predict(config, image_path=args.image, checkpoint_path=checkpoint_path)


if __name__ == "__main__":
    raise SystemExit(main())
