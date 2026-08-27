import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


def read_split(dataset_root, split):
    records = []
    split_dir = dataset_root / split
    for annotation_path in sorted(split_dir.glob("*.json")):
        image_path = split_dir / f"{annotation_path.stem}.jpg"
        data = json.loads(annotation_path.read_text(encoding="utf-8"))
        width = data.get("imageWidth")
        height = data.get("imageHeight")
        if not width or not height:
            with Image.open(image_path) as image:
                width, height = image.size
        boxes = []
        for shape in data.get("shapes", []):
            if shape.get("label") != "cat" or len(shape.get("points", [])) < 2:
                continue
            xs = [point[0] for point in shape["points"]]
            ys = [point[1] for point in shape["points"]]
            boxes.append([min(xs), min(ys), max(xs), max(ys)])
        records.append({"image": image_path, "width": width, "height": height, "boxes": boxes})
    return records


def main(dataset_root, output_dir):
    dataset_root, output_dir = Path(dataset_root), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = {split: read_split(dataset_root, split) for split in ("train", "val", "test")}

    split_counts = {split: len(records) for split, records in datasets.items()}
    box_counts = {split: sum(len(record["boxes"]) for record in records) for split, records in datasets.items()}
    all_records = [record for records in datasets.values() for record in records]
    resolutions = Counter((record["width"], record["height"]) for record in all_records)
    summary = {"image_counts": split_counts, "box_counts": box_counts,
               "total_images": len(all_records), "total_boxes": sum(box_counts.values()),
               "resolution_counts": {f"{w}x{h}": count for (w, h), count in resolutions.items()}}
    (output_dir / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes[0, 0].bar(split_counts.keys(), split_counts.values(), color=["#4C78A8", "#F58518", "#54A24B"])
    axes[0, 0].set(title="Images per Split", ylabel="Images")
    axes[0, 1].bar(box_counts.keys(), box_counts.values(), color=["#4C78A8", "#F58518", "#54A24B"])
    axes[0, 1].set(title="Cat Instances per Split", ylabel="Boxes")
    axes[1, 0].scatter([r["width"] for r in all_records], [r["height"] for r in all_records], alpha=0.35)
    axes[1, 0].set(title="Image Size Distribution", xlabel="Width", ylabel="Height")
    axes[1, 1].hist([r["width"] / r["height"] for r in all_records], bins=25, color="#B279A2")
    axes[1, 1].set(title="Aspect Ratio Distribution", xlabel="Width / Height", ylabel="Images")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "dataset_statistics.png", dpi=180)
    plt.close(fig)

    samples = all_records[:9]
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for axis, record in zip(axes.flat, samples):
        image = Image.open(record["image"]).convert("RGB")
        axis.imshow(image)
        axis.set_title(record["image"].name)
        axis.axis("off")
        for x1, y1, x2, y2 in record["boxes"]:
            axis.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                             fill=False, edgecolor="lime", linewidth=2))
    fig.tight_layout()
    fig.savefig(output_dir / "annotated_samples.png", dpi=160)
    plt.close(fig)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="datastes")
    parser.add_argument("--output-dir", default="outputs/dataset_visualization")
    args = parser.parse_args()
    main(args.dataset_root, args.output_dir)
