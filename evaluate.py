import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_iou
from torchvision.transforms.functional import to_pil_image

from dataset import LabelmeDetectionDataset, collate_fn
from model import build_model
from train import evaluate
from utils import choose_device, load_config
from visualization import plot_training_curves, save_detection_comparison


def main(config_path, checkpoint_path, split):
    config = load_config(config_path)
    device = choose_device(config["device"])
    data_cfg, model_cfg = config["data"], config["model"]
    dataset = LabelmeDetectionDataset(
        data_cfg["dataset_root"], split, data_cfg["class_names"]
    )
    loader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=False,
                        num_workers=data_cfg["num_workers"], collate_fn=collate_fn)
    model = build_model(len(data_cfg["class_names"]) + 1, False,
                        model_cfg["trainable_backbone_layers"], model_cfg["min_size"], model_cfg["max_size"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    eval_cfg = config["evaluation"]
    metrics = evaluate(model, loader, device, eval_cfg["score_threshold"], eval_cfg["iou_threshold"])
    output_dir = Path(checkpoint_path).parent / f"evaluation_{split}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Save the most intuitive error cases: images containing any FP or FN.
    model.eval()
    saved = 0
    with torch.inference_mode():
        for images, targets in loader:
            predictions = model([image.to(device) for image in images])
            for image, target, prediction in zip(images, targets, predictions):
                keep = prediction["scores"].detach().cpu() >= eval_cfg["score_threshold"]
                pred_boxes = prediction["boxes"].detach().cpu()[keep]
                gt_boxes = target["boxes"]
                has_error = len(pred_boxes) != len(gt_boxes)
                if not has_error and len(gt_boxes) and len(pred_boxes):
                    ious = box_iou(pred_boxes, gt_boxes)
                    has_bad_prediction = (ious.max(dim=1).values < eval_cfg["iou_threshold"]).any()
                    has_missed_target = (ious.max(dim=0).values < eval_cfg["iou_threshold"]).any()
                    has_error = bool(has_bad_prediction or has_missed_target)
                if has_error:
                    cpu_prediction = {key: value.detach().cpu() for key, value in prediction.items()}
                    save_detection_comparison(to_pil_image(image), target, cpu_prediction,
                                              output_dir / "error_cases" / f"error_{saved:03d}.jpg",
                                              eval_cfg["score_threshold"])
                    saved += 1
                    if saved >= eval_cfg["max_error_images"]:
                        break
            if saved >= eval_cfg["max_error_images"]:
                break
    plot_training_curves(Path(checkpoint_path).parent / "metrics.csv",
                         output_dir / "training_curves.png")
    print(json.dumps(metrics, indent=2))
    print(f"Visualizations saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--checkpoint", default="outputs/exp001_baseline/best.pt")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.split)
