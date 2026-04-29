#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from common import (
    DEFAULT_DATASET_PATH,
    DEFAULT_DEPTH_SCALE,
    DEFAULT_MODEL_PATH,
    DEFAULT_RAW_TYPE,
    dataset_depth_range,
    default_output_path,
    display_path,
    load_hammer_samples,
    project_path,
    read_gt_depth_and_mask,
    validate_static_inputs,
    write_json,
)


METRIC_FIELDS = [
    "L1",
    "rmse_linear",
    "abs_relative_difference",
    "delta4_acc_105",
    "delta5_acc110",
    "delta1_acc",
]
REPORT_FIELDS = [*METRIC_FIELDS, "eval_valid_ratio"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate PromptDA HAMMER predictions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH), help="PromptDA checkpoint path, used for default output naming")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH), help="HAMMER jsonl path")
    parser.add_argument("--raw-type", default=DEFAULT_RAW_TYPE, choices=["d435", "l515", "tof"], help="Raw depth sensor field")
    parser.add_argument("--output", default=None, help="Prediction output directory")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "mps"], help="Metric device")
    parser.add_argument("--depth-scale", type=float, default=DEFAULT_DEPTH_SCALE, help="GT depth PNG scale factor")
    parser.add_argument("--sample-check-limit", type=int, default=16, help="Number of samples to file-check during dry-run")
    parser.add_argument("--batch-size", type=int, default=32, help="Metric batch size for same-shaped samples")
    parser.add_argument("--limit", type=int, default=0, help="Optional sample limit for debugging; 0 means all samples")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and prediction paths without computing metrics")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    return torch.device(name)


def threshold_percentage(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    d1 = pred / gt
    d2 = gt / pred
    passed = (torch.max(d1, d2) < threshold).float()
    passed[~mask] = 0
    valid_count = mask.sum((-1, -2))
    return torch.sum(passed, (-1, -2)) / valid_count


def compute_metrics_batch(
    pred_np: list[np.ndarray],
    gt_np: list[np.ndarray],
    mask_np: list[np.ndarray],
    device: torch.device,
) -> dict[str, np.ndarray]:
    pred = torch.from_numpy(np.stack(pred_np)).float().to(device)
    gt = torch.from_numpy(np.stack(gt_np)).float().to(device)
    mask = torch.from_numpy(np.stack(mask_np)).bool().to(device)

    diff = pred - gt
    diff[~mask] = 0
    valid_count = mask.sum((-1, -2))

    abs_diff = torch.abs(diff)
    l1 = torch.sum(abs_diff, (-1, -2)) / valid_count
    rmse = torch.sqrt(torch.sum(diff.pow(2), (-1, -2)) / valid_count)

    abs_rel = torch.abs(pred - gt) / gt
    abs_rel[~mask] = 0
    abs_rel = torch.sum(abs_rel, (-1, -2)) / valid_count

    return {
        "L1": l1.detach().cpu().numpy(),
        "rmse_linear": rmse.detach().cpu().numpy(),
        "abs_relative_difference": abs_rel.detach().cpu().numpy(),
        "delta4_acc_105": threshold_percentage(pred, gt, mask, 1.05).detach().cpu().numpy(),
        "delta5_acc110": threshold_percentage(pred, gt, mask, 1.10).detach().cpu().numpy(),
        "delta1_acc": threshold_percentage(pred, gt, mask, 1.25).detach().cpu().numpy(),
    }


def mean_metrics(rows: list[dict]) -> dict[str, float]:
    means = {}
    for field in REPORT_FIELDS:
        values = [float(row[field]) for row in rows if not math.isnan(float(row[field]))]
        means[field] = float(sum(values) / len(values)) if values else float("nan")
    return means


def flush_batch(
    rows: list[dict],
    names: list[str],
    preds: list[np.ndarray],
    gts: list[np.ndarray],
    masks: list[np.ndarray],
    valid_ratios: list[float],
    device: torch.device,
) -> None:
    if not names:
        return
    metrics = compute_metrics_batch(preds, gts, masks, device)
    for index, name in enumerate(names):
        row = {"name": name}
        for field in METRIC_FIELDS:
            row[field] = float(metrics[field][index])
        row["eval_valid_ratio"] = float(valid_ratios[index])
        rows.append(row)
    names.clear()
    preds.clear()
    gts.clear()
    masks.clear()
    valid_ratios.clear()


def run_dry_run(args: argparse.Namespace, output_path: Path) -> int:
    samples = validate_static_inputs(
        args.model_path,
        args.dataset,
        args.raw_type,
        output_path,
        sample_check_limit=args.sample_check_limit,
        require_model=False,
    )
    checked = samples if args.sample_check_limit <= 0 else samples[: args.sample_check_limit]
    missing_predictions = [
        display_path(output_path / f"{sample.name}.npy")
        for sample in checked
        if not (output_path / f"{sample.name}.npy").is_file()
    ]
    print("Evaluation dry-run OK")
    print(f"  output:  {display_path(output_path)}")
    print(f"  samples: {len(samples)}")
    if missing_predictions:
        print("  prediction files not present yet for checked samples:")
        for path in missing_predictions[:10]:
            print(f"    - {path}")
    return 0


def main() -> int:
    args = parse_args()
    model_path = project_path(args.model_path)
    dataset_path = project_path(args.dataset)
    output_path = project_path(args.output) if args.output else default_output_path(model_path, args.raw_type)

    if args.dry_run:
        return run_dry_run(args, output_path)

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if not output_path.is_dir():
        raise FileNotFoundError(f"Prediction output directory does not exist: {display_path(output_path)}")

    samples = load_hammer_samples(dataset_path, args.raw_type)
    if args.limit > 0:
        samples = samples[: args.limit]
    min_depth, max_depth = dataset_depth_range(samples)

    write_json(
        output_path / "eval_args.json",
        {
            **vars(args),
            "model_path": str(model_path),
            "dataset": str(dataset_path),
            "output": str(output_path),
            "min_depth": min_depth,
            "max_depth": max_depth,
            "resolved_samples": len(samples),
        },
    )

    device = resolve_device(args.device)
    rows: list[dict] = []
    names: list[str] = []
    preds: list[np.ndarray] = []
    gts: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    valid_ratios: list[float] = []
    pending_shape = None

    for offset, sample in enumerate(samples, start=1):
        pred_path = output_path / f"{sample.name}.npy"
        if not pred_path.is_file():
            raise FileNotFoundError(f"Missing prediction for {sample.name}: {display_path(pred_path)}")

        pred = np.load(pred_path).astype(np.float32)
        gt, valid_mask = read_gt_depth_and_mask(sample.gt_depth_path, args.depth_scale, min_depth, max_depth)
        if pred.shape != gt.shape:
            raise ValueError(
                f"Prediction shape mismatch for {sample.name}: pred={pred.shape}, gt={gt.shape}"
            )

        gt_valid_count = int(valid_mask.sum())
        if gt_valid_count == 0:
            raise ValueError(f"No valid GT pixels for {sample.name}")

        invalid_pred = ~np.isfinite(pred)
        if invalid_pred.any():
            valid_mask = valid_mask & ~invalid_pred
        if int(valid_mask.sum()) == 0:
            raise ValueError(f"No valid evaluation pixels for {sample.name} after removing invalid predictions")
        valid_ratio = float(valid_mask.sum() / gt_valid_count)

        if pending_shape is not None and pred.shape != pending_shape:
            flush_batch(rows, names, preds, gts, masks, valid_ratios, device)
            pending_shape = None

        pending_shape = pred.shape
        names.append(sample.name)
        preds.append(pred)
        gts.append(gt)
        masks.append(valid_mask)
        valid_ratios.append(valid_ratio)

        if len(names) >= args.batch_size:
            flush_batch(rows, names, preds, gts, masks, valid_ratios, device)
            pending_shape = None

        if offset == 1 or offset % 200 == 0 or offset == len(samples):
            print(f"[{offset}/{len(samples)}] evaluated {sample.name}")

    flush_batch(rows, names, preds, gts, masks, valid_ratios, device)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    all_metrics_path = output_path / f"all_metrics_{timestamp}_False.csv"
    mean_metrics_path = output_path / f"mean_metrics_{timestamp}_False.json"

    with all_metrics_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["name", *REPORT_FIELDS])
        writer.writeheader()
        writer.writerows(rows)

    with mean_metrics_path.open("w", encoding="utf-8") as file:
        file.write(json.dumps([mean_metrics(rows)], ensure_ascii=False) + "\n")

    print(f"Evaluation complete: {display_path(output_path)}")
    print(f"  all metrics:  {display_path(all_metrics_path)}")
    print(f"  mean metrics: {display_path(mean_metrics_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
