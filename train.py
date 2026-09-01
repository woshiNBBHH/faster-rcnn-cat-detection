import argparse
from pathlib import Path

import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import LabelmeDetectionDataset, collate_fn
from metrics import DetectionMetrics
from model import build_model
from utils import append_csv, choose_device, load_config, set_seed
from visualization import plot_training_curves


@torch.inference_mode()
def evaluate(model, loader, device, score_threshold=0.5, iou_threshold=0.5):
    metric = DetectionMetrics(score_threshold, iou_threshold)
    model.eval()
    for images, targets in tqdm(loader, desc="Validating", leave=False):
        images = [image.to(device) for image in images]
        predictions = model(images)
        metric.update(predictions, list(targets))
    return metric.compute()


def main(config_path):
    config = load_config(config_path)
    set_seed(config["seed"])
    device = choose_device(config["device"])
    data_cfg, model_cfg, train_cfg = config["data"], config["model"], config["training"]
    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    train_set = LabelmeDetectionDataset(
        data_cfg["dataset_root"], "train", data_cfg["class_names"],
        train_cfg["horizontal_flip_probability"],
    )
    val_set = LabelmeDetectionDataset(
        data_cfg["dataset_root"], "val", data_cfg["class_names"]
    )
    loader_args = dict(batch_size=train_cfg["batch_size"], num_workers=data_cfg["num_workers"],
                       collate_fn=collate_fn, pin_memory=device.type == "cuda")
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)

    model = build_model(
        len(data_cfg["class_names"]) + 1,
        model_cfg["pretrained"], model_cfg["trainable_backbone_layers"],
        model_cfg["min_size"], model_cfg["max_size"],
        config["evaluation"].get("nms_threshold", 0.5),
    ).to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.SGD(parameters, lr=train_cfg["learning_rate"],
                                momentum=train_cfg["momentum"], weight_decay=train_cfg["weight_decay"])
    scheduler = StepLR(optimizer, step_size=train_cfg["step_size"], gamma=train_cfg["gamma"])
    scaler = torch.amp.GradScaler("cuda", enabled=train_cfg["amp"] and device.type == "cuda")
    writer = SummaryWriter(output_dir / "tensorboard")
    best_map = -1.0

    for epoch in range(1, train_cfg["epochs"] + 1):
        model.train()
        total_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{train_cfg['epochs']}")
        for images, targets in progress:
            images = [image.to(device) for image in images]
            targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                losses = model(images, targets)
                loss = sum(losses.values())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, train_cfg["gradient_clip_norm"])
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        mean_loss = total_loss / len(train_loader)
        metrics = evaluate(model, val_loader, device, config["evaluation"]["score_threshold"],
                           config["evaluation"]["iou_threshold"])
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        map_value = metrics.get("map", 0.0)
        row = {"epoch": epoch, "lr": current_lr, "train_loss": mean_loss,
               "map": map_value, "map_50": metrics.get("map_50", 0.0),
               "mar_100": metrics.get("mar_100", 0.0),
               "tp": metrics["tp"], "fp": metrics["fp"], "fn": metrics["fn"],
               "precision": metrics["precision"], "recall": metrics["recall"],
               "over_detection_rate": metrics["over_detection_rate"],
               "miss_rate": metrics["miss_rate"]}
        append_csv(output_dir / "metrics.csv", row)
        plot_training_curves(output_dir / "metrics.csv", output_dir / "training_curves.png")
        for key, value in row.items():
            if key != "epoch":
                writer.add_scalar(key, value, epoch)

        checkpoint = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "config": config, "metrics": row}
        torch.save(checkpoint, output_dir / "last.pt")
        if map_value > best_map:
            best_map = map_value
            torch.save(checkpoint, output_dir / "best.pt")
        print(row)

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()
    main(args.config)
