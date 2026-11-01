"""Two-stage cat/dog body -> face detection on one image."""

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torchvision.ops import batched_nms
from torchvision.transforms.functional import pil_to_tensor

from model import build_model
from utils import choose_device, load_config


COLORS = {
    "cat": "#2f80ed",
    "dog": "#f2994a",
    "cat_face": "#00b894",
    "dog_face": "#d63031",
}


def load_detector(config_path, checkpoint_path, device):
    config = load_config(config_path)
    model_cfg = config["model"]
    model = build_model(
        len(config["data"]["class_names"]) + 1,
        False,
        model_cfg["trainable_backbone_layers"],
        model_cfg["min_size"],
        model_cfg["max_size"],
        config["evaluation"].get("nms_threshold", 0.5),
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint.get("model_state_dict", checkpoint))
    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model, config


def expand_box(box, image_size, ratio):
    x1, y1, x2, y2 = [float(value) for value in box]
    width, height = image_size
    pad_x = (x2 - x1) * ratio
    pad_y = (y2 - y1) * ratio
    return (
        max(0, int(x1 - pad_x)),
        max(0, int(y1 - pad_y)),
        min(width, int(x2 + pad_x)),
        min(height, int(y2 + pad_y)),
    )


def predict(model, image, device):
    tensor = pil_to_tensor(image).float().div(255.0).to(device)
    with torch.inference_mode():
        return model([tensor])[0]


def class_name(class_names, label):
    index = int(label) - 1
    return class_names[index] if 0 <= index < len(class_names) else str(int(label))


def run_cascade(args):
    device = choose_device(args.device)
    body_model, body_config = load_detector(
        args.body_config, args.body_checkpoint, device
    )
    face_model, face_config = load_detector(
        args.face_config, args.face_checkpoint, device
    )
    image = Image.open(args.image).convert("RGB")
    body_prediction = predict(body_model, image, device)
    body_names = body_config["data"]["class_names"]
    face_names = face_config["data"]["class_names"]

    bodies = []
    face_candidates = []
    for body_box, score, label in zip(
        body_prediction["boxes"].cpu(),
        body_prediction["scores"].cpu(),
        body_prediction["labels"].cpu(),
    ):
        if float(score) < args.body_threshold:
            continue
        species = class_name(body_names, label)
        if species not in {"cat", "dog"}:
            continue
        crop_box = expand_box(body_box, image.size, args.body_expand)
        crop = image.crop(crop_box)
        bodies.append((body_box.tolist(), float(score), species))
        face_prediction = predict(face_model, crop, device)
        for face_box, face_score, face_label in zip(
            face_prediction["boxes"].cpu(),
            face_prediction["scores"].cpu(),
            face_prediction["labels"].cpu(),
        ):
            if float(face_score) < args.face_threshold:
                continue
            face_class = class_name(face_names, face_label)
            if args.require_species_match and face_class != f"{species}_face":
                continue
            x1, y1, x2, y2 = face_box.tolist()
            offset_x, offset_y = crop_box[:2]
            mapped_box = [x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y]
            face_candidates.append(
                (mapped_box, float(face_score), face_class, int(face_label))
            )

    if face_candidates:
        boxes = torch.tensor([item[0] for item in face_candidates], dtype=torch.float32)
        scores = torch.tensor([item[1] for item in face_candidates], dtype=torch.float32)
        labels = torch.tensor([item[3] for item in face_candidates], dtype=torch.int64)
        keep = batched_nms(boxes, scores, labels, args.final_nms)
        faces = [face_candidates[index] for index in keep.tolist()]
    else:
        faces = []

    output = image.copy()
    draw = ImageDraw.Draw(output)
    for box, score, species in bodies:
        draw.rectangle(box, outline=COLORS[species], width=3)
        draw.text((box[0], box[1]), f"{species} {score:.3f}", fill=COLORS[species])
    for box, score, face_class, _ in faces:
        draw.rectangle(box, outline=COLORS.get(face_class, "red"), width=5)
        draw.text(
            (box[0], max(0, box[1] - 14)),
            f"{face_class} {score:.3f}",
            fill=COLORS.get(face_class, "red"),
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.save(output_path)

    if args.face_crop_dir:
        crop_dir = Path(args.face_crop_dir)
        crop_dir.mkdir(parents=True, exist_ok=True)
        for index, (box, score, face_class, _) in enumerate(faces, start=1):
            crop_box = tuple(round(value) for value in box)
            crop = image.crop(crop_box)
            crop.save(crop_dir / f"{Path(args.image).stem}_{index:02d}_{face_class}_{score:.3f}.jpg")

    print(
        f"Saved {output_path}; bodies={len(bodies)}; faces={len(faces)}; device={device}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--body-config", default="configs/cat_dog_baseline.yaml")
    parser.add_argument("--body-checkpoint", required=True)
    parser.add_argument("--face-config", default="configs/cat_dog_face_baseline.yaml")
    parser.add_argument("--face-checkpoint", required=True)
    parser.add_argument("--output", default="outputs/cascade_prediction.jpg")
    parser.add_argument("--face-crop-dir")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--body-threshold", type=float, default=0.5)
    parser.add_argument("--face-threshold", type=float, default=0.5)
    parser.add_argument("--body-expand", type=float, default=0.05)
    parser.add_argument("--final-nms", type=float, default=0.5)
    parser.add_argument(
        "--require-species-match", action=argparse.BooleanOptionalAction, default=True
    )
    run_cascade(parser.parse_args())
