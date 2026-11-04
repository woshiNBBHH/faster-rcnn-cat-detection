"""Convert Oxford-IIIT Pet head ROIs into this project's LabelMe dataset layout."""

import argparse
import json
import os
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


SPECIES_TO_LABEL = {"cat": "cat_face", "dog": "dog_face"}


def transfer(source, destination, mode):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    if mode == "hardlink":
        os.link(source, destination)
    elif mode == "symlink":
        destination.symlink_to(source.resolve())
    else:
        shutil.copy2(source, destination)


def parse_xml(xml_path, images_dir):
    root = ET.parse(xml_path).getroot()
    filename = root.findtext("filename")
    if not filename:
        raise ValueError(f"Missing filename in {xml_path}")
    image_path = images_dir / filename
    if not image_path.is_file():
        raise FileNotFoundError(f"Missing image for {xml_path}: {image_path}")

    with Image.open(image_path) as image:
        actual_width, actual_height = image.size
    xml_width = int(root.findtext("size/width", actual_width))
    xml_height = int(root.findtext("size/height", actual_height))
    if (actual_width, actual_height) != (xml_width, xml_height):
        raise ValueError(
            f"Image/XML size mismatch for {filename}: "
            f"image={(actual_width, actual_height)}, xml={(xml_width, xml_height)}"
        )

    shapes = []
    labels = []
    for obj in root.findall("object"):
        species = (obj.findtext("name") or "").strip().lower()
        if species not in SPECIES_TO_LABEL:
            raise ValueError(f"Unknown species {species!r} in {xml_path}")
        box = obj.find("bndbox")
        if box is None:
            raise ValueError(f"Missing bndbox in {xml_path}")
        x1 = max(0.0, float(box.findtext("xmin")))
        y1 = max(0.0, float(box.findtext("ymin")))
        x2 = min(float(actual_width), float(box.findtext("xmax")))
        y2 = min(float(actual_height), float(box.findtext("ymax")))
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid box in {xml_path}: {(x1, y1, x2, y2)}")
        label = SPECIES_TO_LABEL[species]
        labels.append(label)
        shapes.append(
            {
                "label": label,
                "points": [[x1, y1], [x2, y2]],
                "group_id": None,
                "description": "Oxford-IIIT Pet head ROI",
                "shape_type": "rectangle",
                "flags": {"source": "oxford_iiit_pet"},
                "mask": None,
            }
        )

    if not shapes:
        raise ValueError(f"No cat/dog head boxes in {xml_path}")

    annotation = {
        "version": "5.5.0",
        "flags": {"source": "oxford_iiit_pet"},
        "shapes": shapes,
        "imagePath": filename,
        "imageData": None,
        "imageHeight": actual_height,
        "imageWidth": actual_width,
    }
    return image_path, annotation, labels


def split_samples(samples, train_ratio, val_ratio, rng):
    by_primary_label = defaultdict(list)
    for sample in samples:
        by_primary_label[sample[2][0]].append(sample)

    splits = {"train": [], "val": [], "test": []}
    for label, label_samples in sorted(by_primary_label.items()):
        rng.shuffle(label_samples)
        train_end = int(len(label_samples) * train_ratio)
        val_end = train_end + int(len(label_samples) * val_ratio)
        splits["train"].extend(label_samples[:train_end])
        splits["val"].extend(label_samples[train_end:val_end])
        splits["test"].extend(label_samples[val_end:])
    for samples_in_split in splits.values():
        rng.shuffle(samples_in_split)
    return splits


def main(args):
    source_root = Path(args.source_root)
    images_dir = source_root / "images"
    xml_dir = source_root / "annotations" / "xmls"
    output_root = Path(args.output_root)
    xml_paths = sorted(xml_dir.glob("*.xml"))
    if not xml_paths:
        raise RuntimeError(f"No Oxford XML annotations found in {xml_dir}")

    samples = [parse_xml(xml_path, images_dir) for xml_path in xml_paths]
    splits = split_samples(
        samples, args.train_ratio, args.val_ratio, random.Random(args.seed)
    )

    summary_splits = {}
    for split, split_samples_list in splits.items():
        label_counts = Counter()
        for image_path, annotation, labels in split_samples_list:
            destination_image = output_root / split / image_path.name
            destination_json = output_root / split / f"{image_path.stem}.json"
            transfer(image_path, destination_image, args.mode)
            destination_json.parent.mkdir(parents=True, exist_ok=True)
            destination_json.write_text(
                json.dumps(annotation, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            label_counts.update(labels)
        summary_splits[split] = {
            "images": len(split_samples_list),
            "boxes": dict(label_counts),
        }

    summary = {
        "dataset": "Oxford-IIIT Pet",
        "source_root": str(source_root.resolve()),
        "source_xml_annotations": len(xml_paths),
        "class_names": ["cat_face", "dog_face"],
        "seed": args.seed,
        "mode": args.mode,
        "ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": 1.0 - args.train_ratio - args.val_ratio,
        },
        "splits": summary_splits,
        "license": "CC BY-SA 4.0; image copyright remains with original owners",
        "homepage": "https://www.robots.ox.ac.uk/~vgg/data/pets/",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", default="oxford_pet_face_dataset")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mode", choices=["hardlink", "symlink", "copy"], default="hardlink"
    )
    args = parser.parse_args()
    if args.train_ratio <= 0 or args.val_ratio <= 0 or args.train_ratio + args.val_ratio >= 1:
        parser.error("Ratios must leave positive train, val, and test splits")
    main(args)
