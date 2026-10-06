"""Train and freeze the published SEW 7B-Net on the raw DVS Gesture cache."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset

from spikingjelly.activation_based import functional

from .snn7b import SEW7BNet, load_sew7b_state_dict, parameter_count


class CachedGestureDataset(Dataset):
    """Memory-mapped gesture count frames."""

    def __init__(self, cache: Path, split: str, limit: int = None):
        cache = Path(cache)
        self.metadata = json.loads((cache / "metadata.json").read_text())
        self.frames = np.load(cache / "frames.npy", mmap_mode="r")
        self.labels = np.load(cache / "labels.npy", mmap_mode="r")
        splits = np.load(cache / "splits.npy", mmap_mode="r")
        split_value = {"train": 0, "test": 1}[split]
        self.indices = np.flatnonzero(splits == split_value)
        if limit is not None:
            self.indices = self.indices[:limit]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        cache_index = int(self.indices[index])
        # Copy out of the read-only mmap so PyTorch never receives a
        # non-writable view.  Count values are deliberately not normalized per
        # clip because that would hide event removal by a denoiser.
        frame = np.asarray(self.frames[cache_index], dtype=np.float32).copy()
        return torch.from_numpy(frame), int(self.labels[cache_index]), cache_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--epochs", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--effective-batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lr-step-size", type=int, default=64)
    parser.add_argument("--lr-gamma", type=float, default=0.1)
    parser.add_argument("--t-train", type=int, default=12)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--evaluate-only", type=Path, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def worker_seed(worker_id: int) -> None:
    value = torch.initial_seed() % 2**32
    np.random.seed(value)
    random.seed(value)


def environment_record(device: torch.device) -> dict:
    result = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": str(device),
    }
    if device.type == "cuda":
        result["gpu"] = torch.cuda.get_device_name(device)
        result["gpu_memory_mib"] = torch.cuda.get_device_properties(device).total_memory / 2**20
    return result


def evaluate(model, loader, device, amp_enabled: bool) -> dict:
    model.eval()
    targets = []
    predictions = []
    probabilities = []
    cache_indices = []
    total_loss = 0.0
    total = 0
    with torch.inference_mode():
        for frames, labels, indices in loader:
            frames = frames.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(frames)
                loss = F.cross_entropy(logits, labels)
            probability = logits.float().softmax(dim=1)
            total_loss += float(loss) * len(labels)
            total += len(labels)
            targets.append(labels.cpu().numpy())
            predictions.append(probability.argmax(dim=1).cpu().numpy())
            probabilities.append(probability.cpu().numpy())
            cache_indices.append(indices.numpy())
            functional.reset_net(model)
    target = np.concatenate(targets)
    prediction = np.concatenate(predictions)
    probability = np.concatenate(probabilities)
    index = np.concatenate(cache_indices)
    non_other = target != 10
    prediction_without_other = probability[:, :10].argmax(axis=1)
    return {
        "samples": int(total),
        "loss": total_loss / total,
        "top1": float(np.mean(prediction == target)),
        "macro_f1": float(f1_score(target, prediction, labels=np.arange(11), average="macro")),
        "ten_class_top1_drop_other_logit": float(
            np.mean(prediction_without_other[non_other] == target[non_other])
        ),
        "confusion_matrix": confusion_matrix(
            target, prediction, labels=np.arange(11)
        ).tolist(),
        "target": target.tolist(),
        "prediction": prediction.tolist(),
        "cache_index": index.tolist(),
    }


def atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def atomic_json_write(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.effective_batch_size % args.batch_size:
        raise ValueError("effective-batch-size must be divisible by batch-size")
    accumulation_steps = args.effective_batch_size // args.batch_size
    seed_everything(args.seed)
    device = torch.device(args.device)
    amp_enabled = not args.no_amp and device.type == "cuda"
    args.output.mkdir(parents=True, exist_ok=True)
    train_set = CachedGestureDataset(args.cache, "train", args.max_train_samples)
    test_set = CachedGestureDataset(args.cache, "test", args.max_test_samples)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=worker_seed,
        generator=generator,
        persistent_workers=args.workers > 0,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=worker_seed,
        persistent_workers=args.workers > 0,
    )
    model = SEW7BNet().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    start_epoch = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        load_sew7b_state_dict(model, checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
    if args.evaluate_only is not None:
        checkpoint = torch.load(args.evaluate_only, map_location="cpu", weights_only=False)
        checkpoint_format = load_sew7b_state_dict(model, checkpoint["model"])
        result = evaluate(model, test_loader, device, amp_enabled)
        payload = {
            "checkpoint": str(args.evaluate_only.resolve()),
            "checkpoint_format": checkpoint_format,
            "cache": str(args.cache.resolve()),
            "result": result,
        }
        atomic_json_write(payload, args.output / "evaluation.json")
        print(
            json.dumps(
                {
                    "completed": True,
                    "top1": result["top1"],
                    "macro_f1": result["macro_f1"],
                    "output": str((args.output / "evaluation.json").resolve()),
                },
                indent=2,
            )
        )
        return

    run = {
        "format_version": 1,
        "protocol": {
            "architecture": "SEW 7B-Net, ADD connection",
            "representation": train_set.metadata["representation"],
            "split_by": train_set.metadata.get("split_by", "time"),
            "test_use": "official test evaluated once after the fixed epoch schedule",
            "random_temporal_delete": args.t_train,
            "epochs": args.epochs,
            "micro_batch_size": args.batch_size,
            "effective_batch_size": args.effective_batch_size,
            "accumulation_steps": accumulation_steps,
            "optimizer": "SGD",
            "lr": args.lr,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "lr_step_size": args.lr_step_size,
            "lr_gamma": args.lr_gamma,
            "seed": args.seed,
            "amp": amp_enabled,
        },
        "environment": environment_record(device),
        "model": {"trainable_parameters": parameter_count(model)},
        "data": {"train_samples": len(train_set), "test_samples": len(test_set)},
        "epochs": [],
    }
    atomic_json_write(run, args.output / "run.json")
    started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        correct = 0
        samples = 0
        epoch_started = time.perf_counter()
        for batch_index, (frames, labels, _indices) in enumerate(train_loader):
            frames = frames.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if args.t_train and args.t_train < frames.shape[1]:
                selected = np.sort(
                    np.random.choice(frames.shape[1], args.t_train, replace=False)
                )
                frames = frames[:, selected]
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(frames)
                unscaled_loss = F.cross_entropy(logits, labels)
                # The final accumulation group can contain fewer microbatches
                # (1176 samples leave one 8-sample batch after 73 full groups).
                # Normalize by the actual group size so its update has the same
                # scale as an ordinary SGD update on that smaller last batch.
                group_start = (batch_index // accumulation_steps) * accumulation_steps
                group_size = min(accumulation_steps, len(train_loader) - group_start)
                loss = unscaled_loss / group_size
            scaler.scale(loss).backward()
            should_step = (
                (batch_index + 1) % accumulation_steps == 0
                or batch_index + 1 == len(train_loader)
            )
            if should_step:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            total_loss += float(unscaled_loss) * len(labels)
            correct += int((logits.argmax(dim=1) == labels).sum())
            samples += len(labels)
            functional.reset_net(model)
        scheduler.step()
        record = {
            "epoch": epoch,
            "train_loss": total_loss / samples,
            "train_top1": correct / samples,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.perf_counter() - epoch_started,
        }
        run["epochs"].append(record)
        run["wall_seconds"] = time.perf_counter() - started
        checkpoint = {
            "format_version": 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "protocol": run["protocol"],
            "environment": run["environment"],
        }
        atomic_torch_save(checkpoint, args.output / "checkpoint_latest.pth")
        atomic_json_write(run, args.output / "run.json")
        print(json.dumps(record), flush=True)

    final_checkpoint = torch.load(
        args.output / "checkpoint_latest.pth", map_location="cpu", weights_only=False
    )
    atomic_torch_save(final_checkpoint, args.output / "checkpoint_frozen.pth")
    test_result = evaluate(model, test_loader, device, amp_enabled)
    run["test"] = test_result
    run["wall_seconds"] = time.perf_counter() - started
    atomic_json_write(run, args.output / "run.json")
    print(json.dumps({"completed": True, "test": test_result}, indent=2), flush=True)


if __name__ == "__main__":
    main()
