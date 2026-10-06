"""Evaluate paired DVS Gesture frame conditions with one frozen 7B-Net."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binomtest
from torch.utils.data import DataLoader, Dataset

from .snn7b import (
    DVS_GESTURE_SEMANTIC_TO_RELEASED,
    SEW7BNet,
    load_sew7b_state_dict,
    parameter_count,
)
from .train_dvsgesture import evaluate, seed_everything


class ConditionDataset(Dataset):
    def __init__(self, frames_path: Path, labels: np.ndarray, indices=None):
        self.frames = np.load(frames_path, mmap_mode="r")
        self.labels = np.asarray(labels, dtype=np.int64)
        self.indices = (
            np.arange(len(self.frames), dtype=np.int64)
            if indices is None
            else np.asarray(indices, dtype=np.int64)
        )
        if len(self.indices) != len(self.labels):
            raise ValueError("Condition indices and labels have different lengths")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        frame_index = int(self.indices[index])
        frame = np.asarray(self.frames[frame_index], dtype=np.float32).copy()
        # Return the condition-local index so all prediction arrays have the
        # exact manifest clip order even when raw frames use global indices.
        return torch.from_numpy(frame), int(self.labels[index]), index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--comparisons-only",
        action="store_true",
        help="Add direct method-vs-method paired statistics to an existing output JSON.",
    )
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def paired_accuracy_statistics(
    target: np.ndarray,
    baseline_prediction: np.ndarray,
    method_prediction: np.ndarray,
    subjects: np.ndarray,
    seed: int,
    replicates: int,
) -> dict:
    baseline_correct = baseline_prediction == target
    method_correct = method_prediction == target
    delta = method_correct.astype(np.int8) - baseline_correct.astype(np.int8)
    unique_subjects = np.unique(subjects)
    by_subject = {
        str(subject): {
            "clips": int(np.count_nonzero(subjects == subject)),
            "accuracy_delta": float(np.mean(delta[subjects == subject])),
        }
        for subject in unique_subjects
    }
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(replicates, dtype=np.float64)
    subject_indices = {
        subject: np.flatnonzero(subjects == subject) for subject in unique_subjects
    }
    for index in range(replicates):
        sampled_subjects = rng.choice(
            unique_subjects, size=len(unique_subjects), replace=True
        )
        sampled_indices = np.concatenate(
            [subject_indices[subject] for subject in sampled_subjects]
        )
        bootstrap[index] = float(np.mean(delta[sampled_indices]))
    baseline_only = int(np.count_nonzero(baseline_correct & ~method_correct))
    method_only = int(np.count_nonzero(~baseline_correct & method_correct))
    discordant = baseline_only + method_only
    mcnemar_p = (
        float(binomtest(method_only, discordant, 0.5).pvalue)
        if discordant
        else 1.0
    )
    return {
        "clips": int(len(target)),
        "accuracy_delta": float(np.mean(delta)),
        "cluster_bootstrap_unit": "held-out subject",
        "cluster_bootstrap_replicates": replicates,
        "accuracy_delta_95ci": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "mcnemar_exact_two_sided_p": mcnemar_p,
        "baseline_correct_method_wrong": baseline_only,
        "baseline_wrong_method_correct": method_only,
        "by_subject": by_subject,
    }


def baseline_name(condition_name: str) -> str:
    prefix, _method = condition_name.split("__", 1)
    return f"{prefix}__unfiltered"


def add_direct_method_comparisons(
    results: dict,
    subjects: np.ndarray,
    seed: int,
    replicates: int,
) -> None:
    prefixes = ("clean", "noise_1hz", "noise_5hz", "noise_10hz")
    native_methods = (
        "native3d_float",
        "native3d_float_calibrated",
        "native3d_q8_4",
        "native3d_q8_4_calibrated",
    )
    edmamba_methods = ("edmamba_default", "edmamba_retention95")
    method_pairs = [
        (candidate, reference)
        for candidate in native_methods
        for reference in edmamba_methods
    ] + [
        ("native3d_q8_4", "native3d_float"),
        ("native3d_q8_4_calibrated", "native3d_float_calibrated"),
    ]
    ramet_v2_methods = (
        "ramet_v2_strict",
        "ramet_v2_hybrid",
        "ramet_v2_adapted_hybrid",
        "ramet_v2_random_matched_hybrid",
        "ramet_v2_systematic_matched_hybrid",
    )
    v2_references = (
        "edmamba_default",
        "edmamba_retention95",
        "native3d_float",
        "native3d_q8_4",
        "ramet_v2_random_matched_hybrid",
        "ramet_v2_systematic_matched_hybrid",
    )
    method_pairs += [
        (candidate, reference)
        for candidate in ramet_v2_methods[:3]
        for reference in v2_references
        if candidate != reference
    ]
    direct = {}
    for prefix in prefixes:
        for candidate, reference in method_pairs:
            candidate_name = f"{prefix}__{candidate}"
            reference_name = f"{prefix}__{reference}"
            if (
                candidate_name not in results["conditions"]
                or reference_name not in results["conditions"]
            ):
                continue
            candidate_result = results["conditions"][candidate_name]
            reference_result = results["conditions"][reference_name]
            identifier = f"{candidate_name}_vs_{reference_name}"
            comparison_seed = seed + int.from_bytes(
                identifier.encode("utf-8"), "little"
            ) % 2**32
            direct[identifier] = paired_accuracy_statistics(
                np.asarray(candidate_result["target"]),
                np.asarray(reference_result["prediction"]),
                np.asarray(candidate_result["prediction"]),
                subjects,
                comparison_seed,
                replicates,
            )
    # EDZero study: the two controls the operand claim needs are the ring magnitude at the
    # same retention (does the *operand* matter downstream?) and undirected removal at the
    # same retention (is any gain just sparsification?). Pairs are discovered from the
    # condition names rather than hard-coded, so the retention sweep is covered too.
    edzero_pairs = set()
    for name in results["conditions"]:
        prefix, _, method = name.partition("__")
        if not method.startswith("edzero_r"):
            continue
        suffix = method[len("edzero") :]
        references = [f"magnitude{suffix}", f"random{suffix}", f"edzero_causal{suffix}"]
        # Learned baselines, when a run has appended them: the retention-matched row is the
        # like-for-like ranking comparison, the released row is what a user would get.
        for learned in ("edmamba", "edformer"):
            references += [f"{learned}{suffix}", f"{learned}_released"]
        for reference in references:
            if f"{prefix}__{reference}" in results["conditions"]:
                edzero_pairs.add((name, f"{prefix}__{reference}"))
        # Each swept retention is also compared against the frozen operating point.
        if method != "edzero_r0_28" and f"{prefix}__edzero_r0_28" in results["conditions"]:
            edzero_pairs.add((name, f"{prefix}__edzero_r0_28"))
    for candidate_name, reference_name in sorted(edzero_pairs):
        candidate_result = results["conditions"][candidate_name]
        reference_result = results["conditions"][reference_name]
        identifier = f"{candidate_name}_vs_{reference_name}"
        comparison_seed = seed + int.from_bytes(identifier.encode("utf-8"), "little") % 2**32
        direct[identifier] = paired_accuracy_statistics(
            np.asarray(candidate_result["target"]),
            np.asarray(reference_result["prediction"]),
            np.asarray(candidate_result["prediction"]),
            subjects,
            comparison_seed,
            replicates,
        )

    results["direct_method_comparisons"] = direct
    results["direct_comparison_protocol"] = {
        "delta_sign": "candidate minus reference; positive accuracy favors candidate",
        "bootstrap_unit": "held-out subject",
        "bootstrap_replicates": replicates,
        "paired_clips": True,
    }


def main() -> None:
    args = parse_args()
    if args.comparisons_only:
        if not args.output.is_file():
            raise FileNotFoundError(args.output)
        manifest = json.loads(args.manifest.read_text())
        subjects = np.asarray([clip["subject"] for clip in manifest["clips"]])
        results = json.loads(args.output.read_text())
        add_direct_method_comparisons(
            results, subjects, args.seed, args.bootstrap_replicates
        )
        write_json(args.output, results)
        print(
            json.dumps(
                {
                    "completed": True,
                    "direct_comparisons": len(results["direct_method_comparisons"]),
                }
            ),
            flush=True,
        )
        return
    seed_everything(args.seed)
    device = torch.device(args.device)
    amp_enabled = device.type == "cuda" and not args.no_amp
    manifest = json.loads(args.manifest.read_text())
    labels = np.load(manifest["labels"], mmap_mode="r")
    subjects = np.asarray([clip["subject"] for clip in manifest["clips"]])
    if len(labels) != len(subjects):
        raise ValueError("Manifest label and clip counts differ")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = SEW7BNet().to(device)
    checkpoint_format = load_sew7b_state_dict(model, checkpoint["model"])
    model.eval()
    results = {
        "format_version": 1,
        "protocol": {
            "frozen_checkpoint": str(args.checkpoint.resolve()),
            "classifier_training_protocol": checkpoint.get("protocol"),
            "checkpoint_format": checkpoint_format,
            "semantic_to_released_logit_order": (
                list(DVS_GESTURE_SEMANTIC_TO_RELEASED)
                if checkpoint_format.startswith("official")
                else None
            ),
            "paired_clip_order": True,
            "bootstrap_unit": "held-out subject",
            "bootstrap_replicates": args.bootstrap_replicates,
            "seed": args.seed,
            "test_use": "no checkpoint or threshold selection on held-out clips",
        },
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "model": {"parameters": parameter_count(model)},
        "source_manifest": str(args.manifest.resolve()),
        "conditions": {},
        "comparisons": {},
    }
    started = time.perf_counter()
    condition_items = sorted(
        manifest["conditions"].items(),
        key=lambda item: (item[0].split("__")[0], item[0].split("__")[1] != "unfiltered", item[0]),
    )
    for condition_index, (name, specification) in enumerate(condition_items, start=1):
        dataset = ConditionDataset(
            Path(specification["frames"]), labels, specification.get("indices")
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.workers > 0,
        )
        condition_result = evaluate(model, loader, device, amp_enabled)
        condition_result["specification"] = specification
        results["conditions"][name] = condition_result
        baseline = baseline_name(name)
        if baseline != name and baseline in results["conditions"]:
            baseline_result = results["conditions"][baseline]
            comparison_seed = args.seed + int.from_bytes(
                name.encode("utf-8"), "little"
            ) % 2**32
            results["comparisons"][f"{name}_vs_{baseline}"] = paired_accuracy_statistics(
                np.asarray(condition_result["target"]),
                np.asarray(baseline_result["prediction"]),
                np.asarray(condition_result["prediction"]),
                subjects,
                comparison_seed,
                args.bootstrap_replicates,
            )
        elif (
            baseline == name
            and name != "clean__unfiltered"
            and name.endswith("__unfiltered")
            and "clean__unfiltered" in results["conditions"]
        ):
            clean_result = results["conditions"]["clean__unfiltered"]
            comparison_seed = args.seed + int.from_bytes(
                f"{name}-clean".encode("utf-8"), "little"
            ) % 2**32
            results["comparisons"][f"{name}_vs_clean__unfiltered"] = (
                paired_accuracy_statistics(
                    np.asarray(condition_result["target"]),
                    np.asarray(clean_result["prediction"]),
                    np.asarray(condition_result["prediction"]),
                    subjects,
                    comparison_seed,
                    args.bootstrap_replicates,
                )
            )
        results["wall_seconds"] = time.perf_counter() - started
        write_json(args.output, results)
        print(
            json.dumps(
                {
                    "condition": name,
                    "condition_index": condition_index,
                    "conditions": len(condition_items),
                    "top1": condition_result["top1"],
                    "macro_f1": condition_result["macro_f1"],
                    "elapsed_seconds": results["wall_seconds"],
                }
            ),
            flush=True,
        )
    results["completed"] = True
    add_direct_method_comparisons(
        results, subjects, args.seed, args.bootstrap_replicates
    )
    results["wall_seconds"] = time.perf_counter() - started
    write_json(args.output, results)
    print(json.dumps({"completed": True, "wall_seconds": results["wall_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
