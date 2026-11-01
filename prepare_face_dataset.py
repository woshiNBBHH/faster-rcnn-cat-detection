"""Validate face annotations and create identity-disjoint dataset splits."""

import argparse
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
ALLOWED_LABELS = {"cat_face", "dog_face"}


def parse_identity(stem):
    parts = stem.split("_")
    if len(parts) < 4 or parts[0] not in {"cat", "dog"}:
        raise ValueError(
            f"Expected species_breed_identity_frame filename, got: {stem}"
        )
    return parts[0], "_".join(parts[:-1])


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


def split_species_groups(grouped, train_ratio, val_ratio, rng):
    """Split whole identities while approximating target image counts."""
    items = list(grouped.items())
    rng.shuffle(items)
    total = sum(len(samples) for _, samples in items)
    targets = {
        "train": total * train_ratio,
        "val": total * val_ratio,
        "test": total * (1.0 - train_ratio - val_ratio),
    }
    result = {"train": [], "val": [], "test": []}
    counts = Counter()
    for identity, samples in sorted(items, key=lambda item: len(item[1]), reverse=True):
        split = max(targets, key=lambda name: targets[name] - counts[name])
        result[split].extend(samples)
        counts[split] += len(samples)
    return result


def main(args):
    image_dir = Path(args.image_dir)
    annotation_dir = Path(args.annotation_dir)
    output_root = Path(args.output_root)
    grouped = {"cat": defaultdict(list), "dog": defaultdict(list)}
    box_counts = Counter()
    empty_annotations = 0

    for annotation_path in sorted(annotation_dir.glob("*.json")):
        if annotation_path.name.endswith("summary.json"):
            continue
        image_path = next(
            (
                image_dir / f"{annotation_path.stem}{suffix}"
                for suffix in IMAGE_EXTENSIONS
                if (image_dir / f"{annotation_path.stem}{suffix}").exists()
            ),
            None,
        )
        if image_path is None:
            raise FileNotFoundError(f"No image matches {annotation_path}")

        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        labels = [shape.get("label") for shape in annotation.get("shapes", [])]
        unknown = sorted(set(labels) - ALLOWED_LABELS)
        if unknown:
            raise ValueError(f"Unknown labels in {annotation_path}: {unknown}")
        if not labels:
            empty_annotations += 1
        box_counts.update(labels)

        species, identity = parse_identity(annotation_path.stem)
        expected_label = f"{species}_face"
        wrong_species = [label for label in labels if label != expected_label]
        if wrong_species and not args.allow_mixed_species:
            raise ValueError(
                f"{annotation_path.name} is a {species} source image but contains "
                f"{sorted(set(wrong_species))}; pass --allow-mixed-species for multi-animal images"
            )
        grouped[species][identity].append((image_path, annotation_path, identity))

    if not any(grouped[species] for species in grouped):
        raise RuntimeError("No annotated image/JSON pairs found")

    rng = random.Random(args.seed)
    splits = {"train": [], "val": [], "test": []}
    for species in ("cat", "dog"):
        species_splits = split_species_groups(
            grouped[species], args.train_ratio, args.val_ratio, rng
        )
        for split, samples in species_splits.items():
            splits[split].extend(samples)

    split_identities = {}
    for split, samples in splits.items():
        split_identities[split] = {identity for _, _, identity in samples}
        for image_path, annotation_path, _ in samples:
            transfer(image_path, output_root / split / image_path.name, args.mode)
            transfer(annotation_path, output_root / split / annotation_path.name, args.mode)

    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = split_identities[left] & split_identities[right]
        if overlap:
            raise AssertionError(f"Identity leakage between {left} and {right}: {overlap}")

    summary = {
        "seed": args.seed,
        "mode": args.mode,
        "ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": 1.0 - args.train_ratio - args.val_ratio,
        },
        "splits": {
            split: {
                "images": len(samples),
                "identities": len(split_identities[split]),
                "cat_images": sum(path.stem.startswith("cat_") for path, _, _ in samples),
                "dog_images": sum(path.stem.startswith("dog_") for path, _, _ in samples),
            }
            for split, samples in splits.items()
        },
        "boxes": dict(box_counts),
        "empty_annotations": empty_annotations,
        "class_names": ["cat_face", "dog_face"],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--annotation-dir", required=True)
    parser.add_argument("--output-root", default="cat_dog_face_dataset")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-mixed-species", action="store_true")
    parser.add_argument(
        "--mode", choices=["hardlink", "symlink", "copy"], default="hardlink"
    )
    args = parser.parse_args()
    if args.train_ratio <= 0 or args.val_ratio <= 0 or args.train_ratio + args.val_ratio >= 1:
        parser.error("Ratios must leave positive train, val, and test splits")
    main(args)
