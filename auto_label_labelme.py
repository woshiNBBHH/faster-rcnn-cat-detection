"""Generate Labelme rectangle annotations for cat/dog images with COCO Faster R-CNN.

The generated files are automatic pre-labels. They must be reviewed before they
are used as ground truth, especially for small, occluded, or crowded animals.
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm

from utils import choose_device


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
COCO_CLASSES = {17: "cat", 18: "dog"}


def labelme_shape(label, box, score):
    x1, y1, x2, y2 = [round(float(value), 2) for value in box]
    return {
        "label": label,
        "points": [[x1, y1], [x2, y2]],
        "group_id": None,
        "description": f"auto_score={float(score):.6f}",
        "shape_type": "rectangle",
        "flags": {"auto_labeled": True},
        "mask": None,
    }


def make_annotation(image_path, image, prediction, score_threshold):
    shapes = []
    boxes = prediction["boxes"].detach().cpu()
    scores = prediction["scores"].detach().cpu()
    labels = prediction["labels"].detach().cpu()
    for box, score, label_id in zip(boxes, scores, labels):
        label_id = int(label_id)
        if label_id not in COCO_CLASSES or float(score) < score_threshold:
            continue
        shapes.append(labelme_shape(COCO_CLASSES[label_id], box, score))

    return {
        "version": "5.5.0",
        "flags": {"auto_labeled": True},
        "shapes": shapes,
        "imagePath": image_path.name,
        "imageData": None,
        "imageHeight": image.height,
        "imageWidth": image.width,
    }


def main(args):
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise RuntimeError(f"No supported images found in {image_dir}")

    device = choose_device(args.device)
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    model.roi_heads.nms_thresh = args.nms_threshold
    model.to(device).eval()

    processed = skipped = cat_boxes = dog_boxes = 0
    pending = [
        path for path in image_paths
        if args.overwrite or not (output_dir / f"{path.stem}.json").exists()
    ]
    skipped += len(image_paths) - len(pending)
    progress = tqdm(total=len(pending), desc="Auto-labeling")
    for start in range(0, len(pending), args.batch_size):
        batch_paths = pending[start:start + args.batch_size]
        valid_paths, images, tensors = [], [], []
        for image_path in batch_paths:
            try:
                image = Image.open(image_path).convert("RGB")
            except (OSError, UnidentifiedImageError) as error:
                print(f"Skipping unreadable image {image_path}: {error}")
                skipped += 1
                progress.update(1)
                continue
            valid_paths.append(image_path)
            images.append(image)
            tensors.append(pil_to_tensor(image).float().div(255.0).to(device))

        if not tensors:
            continue
        with torch.inference_mode():
            predictions = model(tensors)
        for image_path, image, prediction in zip(valid_paths, images, predictions):
            annotation = make_annotation(image_path, image, prediction, args.score_threshold)
            for shape in annotation["shapes"]:
                cat_boxes += shape["label"] == "cat"
                dog_boxes += shape["label"] == "dog"
            (output_dir / f"{image_path.stem}.json").write_text(
                json.dumps(annotation, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            processed += 1
            progress.update(1)
    progress.close()

    summary = {
        "images_found": len(image_paths),
        "images_processed": processed,
        "images_skipped": skipped,
        "cat_boxes": cat_boxes,
        "dog_boxes": dog_boxes,
        "score_threshold": args.score_threshold,
        "nms_threshold": args.nms_threshold,
        "warning": "Automatic labels require human review before training.",
    }
    (output_dir / "auto_label_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--score-threshold", type=float, default=0.8)
    parser.add_argument("--nms-threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    main(parser.parse_args())
