"""Create a reproducible, source-balanced 4,000-image cat/dog seed set.

The source dataset uses filename prefixes (coco2017, lvis, openimages, voc).
By default this script samples 1,000 images from each source with seed 42 and
assigns 800/100/100 images per source to train/val/test.
"""

import argparse
import csv
import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DEFAULT_SOURCES = ("coco2017", "lvis", "openimages", "voc")


def source_from_name(path):
    return path.stem.split("_", 1)[0].lower()


def place_file(source, destination, mode):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        return
    if mode == "copy":
        shutil.copy2(source, destination)
    elif mode == "hardlink":
        os.link(source, destination)
    elif mode == "symlink":
        destination.symlink_to(source.resolve())


def main(args):
    image_dir = Path(args.image_dir)
    output_root = Path(args.output_root)
    manifest_dir = output_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    grouped = {source: [] for source in args.sources}
    for path in sorted(image_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            source = source_from_name(path)
            if source in grouped:
                grouped[source].append(path)

    rng = random.Random(args.seed)
    rows = []
    for source in args.sources:
        candidates = grouped[source]
        if len(candidates) < args.samples_per_source:
            raise RuntimeError(
                f"{source} has {len(candidates)} images, fewer than "
                f"the requested {args.samples_per_source}"
            )
        selected = rng.sample(candidates, args.samples_per_source)
        rng.shuffle(selected)
        train_end = int(args.samples_per_source * args.train_ratio)
        val_end = train_end + int(args.samples_per_source * args.val_ratio)
        for index, image_path in enumerate(selected):
            split = "train" if index < train_end else "val" if index < val_end else "test"
            rows.append(
                {
                    "filename": image_path.name,
                    "source": source,
                    "split": split,
                    "relative_path": f"images/{image_path.name}",
                }
            )

    rows.sort(key=lambda row: (row["split"], row["source"], row["filename"]))
    csv_path = manifest_dir / "cat_dog_initial_4000.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    if args.mode != "manifest":
        for row in rows:
            source_path = image_dir / row["filename"]
            destination = output_root / "images" / row["split"] / row["filename"]
            place_file(source_path, destination, args.mode)

    summary = {
        "seed": args.seed,
        "total": len(rows),
        "samples_per_source": dict(Counter(row["source"] for row in rows)),
        "split_counts": dict(Counter(row["split"] for row in rows)),
        "mode": args.mode,
        "manifest": str(csv_path),
    }
    (manifest_dir / "cat_dog_initial_4000_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output-root", default="cat_dog_initial_4000")
    parser.add_argument("--sources", nargs="+", default=list(DEFAULT_SOURCES))
    parser.add_argument("--samples-per-source", type=int, default=1000)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mode",
        choices=["manifest", "symlink", "hardlink", "copy"],
        default="manifest",
        help="manifest writes only the CSV; other modes also materialize images",
    )
    parsed = parser.parse_args()
    if parsed.train_ratio <= 0 or parsed.val_ratio < 0:
        parser.error("train-ratio must be positive and val-ratio cannot be negative")
    if parsed.train_ratio + parsed.val_ratio >= 1:
        parser.error("ratios must leave a positive test split")
    main(parsed)
