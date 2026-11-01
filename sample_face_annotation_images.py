"""Select an identity-diverse batch of cat/dog images for face annotation."""

import argparse
import csv
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_identity(path):
    """Return (species, identity) for species_breed_identity_frame filenames."""
    parts = path.stem.split("_")
    if len(parts) < 4 or parts[0] not in {"cat", "dog"}:
        return None
    return parts[0], "_".join(parts[:-1])


def transfer(source, destination, mode):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")
    if mode == "hardlink":
        os.link(source, destination)
    elif mode == "symlink":
        destination.symlink_to(source.resolve())
    else:
        shutil.copy2(source, destination)


def choose_balanced(groups, target_count, max_per_identity, rng):
    group_items = []
    for identity, paths in groups.items():
        paths = list(paths)
        rng.shuffle(paths)
        group_items.append((identity, paths[:max_per_identity]))
    rng.shuffle(group_items)

    selected = []
    depth = 0
    while len(selected) < target_count:
        added = False
        for identity, paths in group_items:
            if depth < len(paths):
                selected.append((identity, paths[depth]))
                added = True
                if len(selected) == target_count:
                    break
        if not added:
            break
        depth += 1
    return selected


def main(args):
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f"Output directory is not empty: {output_dir}. Use a new directory."
        )

    groups = {"cat": defaultdict(list), "dog": defaultdict(list)}
    skipped_names = 0
    iterator = image_dir.rglob("*") if args.recursive else image_dir.iterdir()
    for path in sorted(iterator):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        parsed = parse_identity(path)
        if parsed is None:
            skipped_names += 1
            continue
        species, identity = parsed
        groups[species][identity].append(path)

    rng = random.Random(args.seed)
    requests = {"cat": args.cat_count, "dog": args.dog_count}
    selections = {}
    for species, count in requests.items():
        selections[species] = choose_balanced(
            groups[species], count, args.max_per_identity, rng
        )
        if len(selections[species]) < count:
            raise RuntimeError(
                f"Requested {count} {species} images, but only "
                f"{len(selections[species])} satisfy max_per_identity={args.max_per_identity}"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for species in ("cat", "dog"):
        for identity, source in selections[species]:
            destination = output_dir / source.name
            transfer(source, destination, args.mode)
            rows.append(
                {
                    "species": species,
                    "identity": identity,
                    "source": str(source.resolve()),
                    "annotation_image": destination.name,
                    "annotation_status": "pending",
                }
            )

    manifest_path = output_dir / "annotation_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "image_dir": str(image_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "seed": args.seed,
        "mode": args.mode,
        "max_per_identity": args.max_per_identity,
        "selected": {species: len(items) for species, items in selections.items()},
        "available_identities": {
            species: len(species_groups) for species, species_groups in groups.items()
        },
        "skipped_unrecognized_filenames": skipped_names,
        "labels": ["cat_face", "dog_face"],
    }
    (output_dir / "sampling_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output-dir", default="face_annotation_batch_v1")
    parser.add_argument("--cat-count", type=int, default=500)
    parser.add_argument("--dog-count", type=int, default=500)
    parser.add_argument("--max-per-identity", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument(
        "--mode", choices=["hardlink", "symlink", "copy"], default="hardlink"
    )
    args = parser.parse_args()
    if args.cat_count < 0 or args.dog_count < 0 or args.max_per_identity <= 0:
        parser.error("Counts must be non-negative and max-per-identity must be positive")
    main(args)
