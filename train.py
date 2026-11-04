import argparse
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import LabelmeDetectionDataset, collate_fn
from metrics import DetectionMetrics
from model import build_model
from utils import append_csv, choose_device, load_config, set_seed
from visualization import plot_training_curves


def setup_distributed(configured_device):
    """Initialize torchrun/DDP when WORLD_SIZE is greater than one."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1

    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training requires CUDA GPUs")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        # Rank 0 performs validation/checkpoint I/O while the other ranks wait.
        # The previous 10-minute timeout was too short for Faster R-CNN COCO
        # evaluation plus the first checkpoint write on Kaggle.
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(hours=1),
        )
        rank = dist.get_rank()
        device = torch.device("cuda", local_rank)
    else:
        local_rank = 0
        rank = 0
        device = choose_device(configured_device)

    return distributed, rank, local_rank, world_size, device


def is_main_process(rank):
    return rank == 0


def unwrap_model(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


def load_initial_checkpoint(model, checkpoint_path):
    """Load model weights for fine-tuning without restoring optimizer state."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Initial checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint.get("model_state_dict", checkpoint))
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint format: {checkpoint_path}")

    # Accept checkpoints saved from either a plain model or a DDP wrapper.
    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)
    return checkpoint.get("epoch") if isinstance(checkpoint, dict) else None


@torch.inference_mode()
def evaluate(model, loader, device, score_threshold=0.5, iou_threshold=0.5):
    metric = DetectionMetrics(score_threshold, iou_threshold)
    model.eval()
    for images, targets in tqdm(loader, desc="Validating", leave=True):
        images = [image.to(device, non_blocking=True) for image in images]
        predictions = model(images)
        metric.update(predictions, list(targets))
    return metric.compute()


def main(config_path):
    config = load_config(config_path)
    distributed, rank, local_rank, world_size, device = setup_distributed(config["device"])
    main_process = is_main_process(rank)

    try:
        # Different augmentation RNG streams per process; DDP synchronizes model weights.
        set_seed(config["seed"] + rank)
        data_cfg = config["data"]
        model_cfg = config["model"]
        train_cfg = config["training"]
        output_dir = Path(train_cfg["output_dir"])

        if main_process:
            output_dir.mkdir(parents=True, exist_ok=True)
        if distributed:
            dist.barrier()

        train_set = LabelmeDetectionDataset(
            data_cfg["dataset_root"],
            "train",
            data_cfg["class_names"],
            train_cfg["horizontal_flip_probability"],
        )
        train_sampler = (
            DistributedSampler(
                train_set,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=config["seed"],
            )
            if distributed
            else None
        )

        # num_workers and batch_size are per process/GPU under DDP.
        loader_args = dict(
            batch_size=train_cfg["batch_size"],
            num_workers=data_cfg["num_workers"],
            collate_fn=collate_fn,
            pin_memory=device.type == "cuda",
            persistent_workers=data_cfg["num_workers"] > 0,
        )
        train_loader = DataLoader(
            train_set,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            **loader_args,
        )

        # Validation runs once on rank 0. Calling the unwrapped model avoids a
        # DDP buffer synchronization while the other rank waits at the barrier.
        val_loader = None
        if main_process:
            val_set = LabelmeDetectionDataset(
                data_cfg["dataset_root"], "val", data_cfg["class_names"]
            )
            val_loader = DataLoader(val_set, shuffle=False, **loader_args)

        model = build_model(
            len(data_cfg["class_names"]) + 1,
            model_cfg["pretrained"],
            model_cfg["trainable_backbone_layers"],
            model_cfg["min_size"],
            model_cfg["max_size"],
            config["evaluation"].get("nms_threshold", 0.5),
        )

        initial_checkpoint = train_cfg.get("initial_checkpoint")
        if initial_checkpoint:
            checkpoint_epoch = load_initial_checkpoint(model, initial_checkpoint)
            if main_process:
                epoch_text = f" (epoch {checkpoint_epoch})" if checkpoint_epoch is not None else ""
                print(f"Loaded initial checkpoint: {initial_checkpoint}{epoch_text}")

        model.to(device)
        if distributed:
            model = DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
            )

        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.SGD(
            parameters,
            lr=train_cfg["learning_rate"],
            momentum=train_cfg["momentum"],
            weight_decay=train_cfg["weight_decay"],
        )
        scheduler = StepLR(
            optimizer,
            step_size=train_cfg["step_size"],
            gamma=train_cfg["gamma"],
        )
        scaler = torch.amp.GradScaler(
            "cuda", enabled=train_cfg["amp"] and device.type == "cuda"
        )
        writer = SummaryWriter(output_dir / "tensorboard") if main_process else None
        best_map = -1.0

        if main_process:
            print(
                f"Training on {world_size} GPU(s); "
                f"batch_size_per_gpu={train_cfg['batch_size']}; "
                f"global_batch_size={train_cfg['batch_size'] * world_size}; "
                f"train_images={len(train_set)}"
            )

        for epoch in range(1, train_cfg["epochs"] + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            model.train()
            total_loss = 0.0
            progress = tqdm(
                train_loader,
                desc=f"Epoch {epoch}/{train_cfg['epochs']}",
                disable=not main_process,
            )
            for images, targets in progress:
                images = [image.to(device, non_blocking=True) for image in images]
                targets = [
                    {key: value.to(device, non_blocking=True) for key, value in target.items()}
                    for target in targets
                ]
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(
                    device_type=device.type, enabled=scaler.is_enabled()
                ):
                    losses = model(images, targets)
                    loss = sum(losses.values())
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    parameters, train_cfg["gradient_clip_norm"]
                )
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item()
                if main_process:
                    progress.set_postfix(loss=f"{loss.item():.4f}")

            # Report a true mean across every rank, including padded sampler items.
            loss_stats = torch.tensor(
                [total_loss, len(train_loader)], dtype=torch.float64, device=device
            )
            if distributed:
                dist.all_reduce(loss_stats, op=dist.ReduceOp.SUM)
            mean_loss = (loss_stats[0] / loss_stats[1]).item()

            metrics = None
            if main_process:
                print(f"Epoch {epoch}: validation started", flush=True)
                metrics = evaluate(
                    unwrap_model(model),
                    val_loader,
                    device,
                    config["evaluation"]["score_threshold"],
                    config["evaluation"]["iou_threshold"],
                )
                print(f"Epoch {epoch}: validation finished", flush=True)

            # Rank 1 waits while rank 0 validates, then both advance the scheduler.
            if distributed:
                dist.barrier()
            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step()

            if main_process:
                map_value = metrics.get("map", 0.0)
                row = {
                    "epoch": epoch,
                    "lr": current_lr,
                    "train_loss": mean_loss,
                    "map": map_value,
                    "map_50": metrics.get("map_50", 0.0),
                    "mar_100": metrics.get("mar_100", 0.0),
                    "tp": metrics["tp"],
                    "fp": metrics["fp"],
                    "fn": metrics["fn"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "over_detection_rate": metrics["over_detection_rate"],
                    "miss_rate": metrics["miss_rate"],
                }
                append_csv(output_dir / "metrics.csv", row)
                for key, value in row.items():
                    if key != "epoch":
                        writer.add_scalar(key, value, epoch)

                # Save the unwrapped state dict so checkpoints remain compatible
                # with single-GPU inference and evaluation.
                checkpoint = {
                    "epoch": epoch,
                    "model": unwrap_model(model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "config": config,
                    "metrics": row,
                    "world_size": world_size,
                    "global_batch_size": train_cfg["batch_size"] * world_size,
                }
                print(f"Epoch {epoch}: saving last.pt", flush=True)
                torch.save(checkpoint, output_dir / "last.pt")
                if map_value > best_map:
                    best_map = map_value
                    print(f"Epoch {epoch}: saving best.pt", flush=True)
                    torch.save(checkpoint, output_dir / "best.pt")
                print(row, flush=True)

            # Do not let rank 1 enter the next DDP forward pass while rank 0 saves.
            if distributed:
                dist.barrier()

        if main_process:
            # Plot once after training instead of making the other DDP rank wait
            # for Matplotlib after every epoch.
            plot_training_curves(
                output_dir / "metrics.csv", output_dir / "training_curves.png"
            )

        if writer is not None:
            writer.close()
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    # Accepted for compatibility with older torchrun versions. LOCAL_RANK from
    # the environment remains the source of truth.
    parser.add_argument("--local-rank", "--local_rank", type=int, default=0)
    args = parser.parse_args()
    main(args.config)
