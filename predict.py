import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torchvision.transforms.functional import pil_to_tensor

from model import build_model
from utils import choose_device, load_config


def main(config_path, checkpoint_path, image_path, output_path):
    config = load_config(config_path)
    device = choose_device(config["device"])
    model_cfg = config["model"]
    model = build_model(len(config["data"]["class_names"]) + 1, False,
                        model_cfg["trainable_backbone_layers"], model_cfg["min_size"], model_cfg["max_size"],
                        config["evaluation"].get("nms_threshold", 0.5))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()

    image = Image.open(image_path).convert("RGB")
    tensor = pil_to_tensor(image).float().div(255.0).to(device)
    with torch.inference_mode():
        prediction = model([tensor])[0]
    draw = ImageDraw.Draw(image)
    threshold = config["evaluation"]["score_threshold"]
    class_names = config["data"]["class_names"]
    for box, score, label in zip(
        prediction["boxes"].cpu(), prediction["scores"].cpu(), prediction["labels"].cpu()
    ):
        if score < threshold:
            continue
        label_index = int(label) - 1
        class_name = class_names[label_index] if 0 <= label_index < len(class_names) else str(int(label))
        draw.rectangle(box.tolist(), outline="red", width=3)
        draw.text((box[0].item(), box[1].item()), f"{class_name} {score:.3f}", fill="red")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--checkpoint", default="outputs/exp001_baseline/best.pt")
    parser.add_argument("--output", default="outputs/prediction.jpg")
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.image, args.output)
