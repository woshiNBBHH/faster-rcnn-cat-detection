import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor


class LabelmeDetectionDataset(Dataset):
    """Read paired images and Labelme JSON files from one split."""

    def __init__(self, dataset_root, split, class_names, flip_probability=0.0):
        # Each split contains paired files with the same stem, such as image001.jpg + image001.json.
        self.split_dir = Path(dataset_root) / split
        self.flip_probability = flip_probability
        self.class_to_id = {name: index + 1 for index, name in enumerate(class_names)}
        self.samples = []

        image_extensions = (".jpg", ".jpeg", ".png")
        for annotation_path in sorted(self.split_dir.glob("*.json")):
            image_path = next(
                (self.split_dir / f"{annotation_path.stem}{suffix}" for suffix in image_extensions
                 if (self.split_dir / f"{annotation_path.stem}{suffix}").exists()),
                None,
            )
            if image_path is None:
                expected = ", ".join(f"{annotation_path.stem}{suffix}" for suffix in image_extensions)
                raise FileNotFoundError(f"No matching image for {annotation_path}; expected one of: {expected}")
            self.samples.append((image_path, annotation_path))

        if not self.samples:
            raise RuntimeError(f"No paired samples found in {self.split_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, annotation_path = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        with annotation_path.open("r", encoding="utf-8") as file:
            annotation = json.load(file)

        boxes, labels = [], []
        for shape in annotation.get("shapes", []):
            label = shape.get("label")
            if label not in self.class_to_id:
                continue
            points = shape.get("points", [])
            shape_type = shape.get("shape_type", "polygon")
            if shape_type not in {"rectangle", "oriented_rectangle", "polygon"} or len(points) < 2:
                continue

            xs = [float(point[0]) for point in points]
            ys = [float(point[1]) for point in points]
            x1, y1 = max(0.0, min(xs)), max(0.0, min(ys))
            x2, y2 = min(float(width), max(xs)), min(float(height), max(ys))
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(self.class_to_id[label])

        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_tensor = pil_to_tensor(image).float().div(255.0)

        if self.flip_probability and random.random() < self.flip_probability:
            image_tensor = image_tensor.flip(-1)
            if len(boxes):
                old_x1 = boxes[:, 0].clone()
                old_x2 = boxes[:, 2].clone()
                boxes[:, 0] = width - old_x2
                boxes[:, 2] = width - old_x1

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor(index, dtype=torch.int64),
            "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
            "iscrowd": torch.zeros(len(boxes), dtype=torch.int64),
        }
        return image_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))
