import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_iou
from torchvision.transforms.functional import to_pil_image

from dataset import LabelmeDetectionDataset, collate_fn
from model import build_model
from utils import choose_device, load_config


def match_predictions(prediction, target, score_threshold, iou_threshold):
    """Greedily match predictions to GT boxes and return TP/FP/FN indices."""
    keep = prediction["scores"].detach().cpu() >= score_threshold
    boxes = prediction["boxes"].detach().cpu()[keep]
    scores = prediction["scores"].detach().cpu()[keep]
    labels = prediction["labels"].detach().cpu()[keep]
    gt_boxes = target["boxes"].detach().cpu()
    gt_labels = target["labels"].detach().cpu()

    order = scores.argsort(descending=True)
    matched_gt = set()
    tp_indices, fp_indices = [], []

    for pred_index in order.tolist():
        candidates = [
            index for index in range(len(gt_boxes))
            if index not in matched_gt and gt_labels[index] == labels[pred_index]
        ]
        if not candidates:
            fp_indices.append(pred_index)
            continue

        ious = box_iou(boxes[pred_index].unsqueeze(0), gt_boxes[candidates])[0]
        best_iou, best_offset = ious.max(dim=0)
        if float(best_iou) >= iou_threshold:
            matched_gt.add(candidates[int(best_offset)])
            tp_indices.append(pred_index)
        else:
            fp_indices.append(pred_index)

    fn_indices = [index for index in range(len(gt_boxes)) if index not in matched_gt]
    return boxes, scores, tp_indices, fp_indices, fn_indices


def add_box(axis, box, color, text):
    x1, y1, x2, y2 = [float(value) for value in box]
    axis.add_patch(patches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        fill=False, edgecolor=color, linewidth=2.5,
    ))
    axis.text(
        x1, max(0, y1 - 3), text, color="white", fontsize=9,
        bbox={"facecolor": color, "alpha": 0.8, "pad": 2},
    )


def save_error_visualization(image, target, boxes, scores, tp_indices,
                             fp_indices, fn_indices, image_name, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].imshow(image)
    axes[0].set_title(f"Ground Truth: {image_name} | FN={len(fn_indices)}")
    axes[0].axis("off")
    for index, box in enumerate(target["boxes"]):
        is_fn = index in fn_indices
        add_box(axes[0], box, "orange" if is_fn else "lime", "FN" if is_fn else "GT")

    axes[1].imshow(image)
    axes[1].set_title(
        f"Prediction: {image_name} | TP={len(tp_indices)} FP={len(fp_indices)}"
    )
    axes[1].axis("off")
    for index in tp_indices:
        add_box(axes[1], boxes[index], "lime", f"TP {float(scores[index]):.3f}")
    for index in fp_indices:
        add_box(axes[1], boxes[index], "red", f"FP {float(scores[index]):.3f}")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)


def main(args):
    config = load_config(args.config)
    data_cfg = config["data"]
    model_cfg = config["model"]
    eval_cfg = config["evaluation"]
    score_threshold = (
        args.score_threshold
        if args.score_threshold is not None
        else eval_cfg["score_threshold"]
    )
    iou_threshold = (
        args.iou_threshold
        if args.iou_threshold is not None
        else eval_cfg["iou_threshold"]
    )

    device = choose_device(config["device"])
    dataset = LabelmeDetectionDataset(
        data_cfg["dataset_root"], args.split, data_cfg["class_names"]
    )
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=data_cfg["num_workers"], collate_fn=collate_fn,
    )
    model = build_model(
        len(data_cfg["class_names"]) + 1,
        False,
        model_cfg["trainable_backbone_layers"],
        model_cfg["min_size"],
        model_cfg["max_size"],
    )
    if "nms_threshold" in eval_cfg:
        model.roi_heads.nms_thresh = float(eval_cfg["nms_threshold"])
    load_checkpoint(model, args.checkpoint)
    model.to(device).eval()

    output_dir = Path(args.output_dir) if args.output_dir else (
        Path(args.checkpoint).parent / f"fp_fn_{args.split}"
    )
    for folder in ("fp", "fn", "fp_and_fn"):
        (output_dir / folder).mkdir(parents=True, exist_ok=True)

    rows = []
    total_fp = total_fn = error_images = 0
    with torch.inference_mode():
        for index, (images, targets) in enumerate(loader):
            image = images[0]
            target = targets[0]
            prediction = model([image.to(device)])[0]
            boxes, scores, tp_indices, fp_indices, fn_indices = match_predictions(
                prediction, target, score_threshold, iou_threshold
            )
            fp_count, fn_count = len(fp_indices), len(fn_indices)
            if not fp_count and not fn_count:
                continue

            image_name = dataset.samples[index][0].name
            category = "fp_and_fn" if fp_count and fn_count else ("fp" if fp_count else "fn")
            output_name = f"{Path(image_name).stem}__FP{fp_count}_FN{fn_count}.jpg"
            save_error_visualization(
                to_pil_image(image), target, boxes, scores,
                tp_indices, fp_indices, fn_indices, image_name,
                output_dir / category / output_name,
            )
            rows.append({
                "image_name": image_name,
                "category": category,
                "gt_count": len(target["boxes"]),
                "prediction_count": len(boxes),
                "tp": len(tp_indices),
                "fp": fp_count,
                "fn": fn_count,
                "visualization": f"{category}/{output_name}",
            })
            total_fp += fp_count
            total_fn += fn_count
            error_images += 1

    rows.sort(key=lambda row: (-(row["fp"] + row["fn"]), row["image_name"]))
    with (output_dir / "error_images.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()) if rows else [
            "image_name", "category", "gt_count", "prediction_count",
            "tp", "fp", "fn", "visualization",
        ])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "split": args.split,
        "score_threshold": score_threshold,
        "iou_threshold": iou_threshold,
        "total_images": len(dataset),
        "error_images": error_images,
        "total_fp": total_fp,
        "total_fn": total_fn,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"FP/FN image list: {output_dir / 'error_images.csv'}")
    print(f"Visualizations: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize and list FP/FN images.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument("--iou-threshold", type=float, default=None)
    parser.add_argument("--output-dir", default=None)
    main(parser.parse_args())
