import csv
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_training_curves(csv_path, output_path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return
    with csv_path.open("r", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        return

    epochs = [int(row["epoch"]) for row in rows]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(epochs, [float(row["train_loss"]) for row in rows], marker="o")
    axes[0].set(title="Training Loss", xlabel="Epoch", ylabel="Loss")
    axes[1].plot(epochs, [float(row["map"]) for row in rows], marker="o", label="mAP@[.5:.95]")
    axes[1].plot(epochs, [float(row["map_50"]) for row in rows], marker="o", label="mAP@0.5")
    axes[1].set(title="mAP Curve", xlabel="Epoch", ylabel="mAP", ylim=(0, 1))
    axes[1].legend()
    axes[2].plot(epochs, [float(row["over_detection_rate"]) for row in rows], marker="o", label="Over-detection")
    axes[2].plot(epochs, [float(row["miss_rate"]) for row in rows], marker="o", label="Miss rate")
    axes[2].set(title="Detection Error Rates", xlabel="Epoch", ylabel="Rate", ylim=(0, 1))
    axes[2].legend()
    for axis in axes:
        axis.grid(alpha=0.3)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _label_name(label, class_names):
    index = int(label) - 1
    return class_names[index] if class_names and 0 <= index < len(class_names) else str(int(label))


def save_detection_comparison(image, target, prediction, output_path,
                              score_threshold=0.5, class_names=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    panels = [(axes[0], target["boxes"], None, target.get("labels"), "Ground Truth"),
              (axes[1], prediction["boxes"], prediction["scores"],
               prediction.get("labels"), "Prediction")]
    for axis, boxes, scores, labels, title in panels:
        axis.imshow(image)
        axis.set_title(title)
        axis.axis("off")
        for index, box in enumerate(boxes):
            if scores is not None and float(scores[index]) < score_threshold:
                continue
            x1, y1, x2, y2 = [float(value) for value in box]
            color = "lime" if scores is None else "red"
            axis.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                             fill=False, edgecolor=color, linewidth=2))
            if labels is not None:
                name = _label_name(labels[index], class_names)
                text = name if scores is None else f"{name} {float(scores[index]):.3f}"
                axis.text(x1, y1, text, color="white",
                          bbox={"facecolor": "red", "alpha": 0.7, "pad": 2})
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
