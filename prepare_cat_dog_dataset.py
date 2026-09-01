"""Create deterministic train/val/test splits from images and Labelme JSON files."""

import argparse
import json
import os
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


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


def main(args):
    image_dir = Path(args.image_dir)
    annotation_dir = Path(args.annotation_dir)
    output_root = Path(args.output_root)
    samples = []
    for annotation_path in sorted(annotation_dir.glob("*.json")):
        if annotation_path.name == "auto_label_summary.json":
            continue
        image_path = next(
            (image_dir / f"{annotation_path.stem}{suffix}" for suffix in IMAGE_EXTENSIONS
             if (image_dir / f"{annotation_path.stem}{suffix}").exists()),
            None,
        )
        if image_path is not None:
            samples.append((image_path, annotation_path))

    if not samples:
        raise RuntimeError("No image/JSON pairs found")
    random.Random(args.seed).shuffle(samples)
    train_end = int(len(samples) * args.train_ratio)
    val_end = train_end + int(len(samples) * args.val_ratio)
    groups = {
        "train": samples[:train_end],
        "val": samples[train_end:val_end],
        "test": samples[val_end:],
    }
    for split, pairs in groups.items():
        for image_path, annotation_path in pairs:
            transfer(image_path, output_root / split / image_path.name, args.mode)
            transfer(annotation_path, output_root / split / annotation_path.name, args.mode)

    summary = {split: len(pairs) for split, pairs in groups.items()}
    summary.update({"total": len(samples), "seed": args.seed, "mode": args.mode})
    (output_root / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--annotation-dir", required=True)
    parser.add_argument("--output-root", default="cat_dog_dataset_53337")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["hardlink", "symlink", "copy"], default="hardlink")
    args = parser.parse_args()
    if args.train_ratio <= 0 or args.val_ratio < 0 or args.train_ratio + args.val_ratio >= 1:
        parser.error("Ratios must leave a positive test split")
    main(args)
