#!/usr/bin/env python3
"""Run PromptDA iBims inference on selected difficulties, then official eval."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for path in (PROJECT_ROOT, SCRIPT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from eval_official import prepare_workspace, resolve_root, run_official_eval  # noqa: E402
from infer_to_mat import (  # noqa: E402
    SYNTHETIC_RAW_DIR_NAME,
    infer_manifest,
    load_model,
    sanitize_model_name,
    validate_common_args,
    write_infer_args,
)


ALL_LEVELS = ("easy", "medium", "hard", "extreme")
RESULT_METRIC_KEYS = [
    "rel",
    "sq_rel",
    "rms",
    "log10",
    "thr1",
    "thr2",
    "thr3",
    "dde_0",
    "dde_p",
    "dde_m",
    "pe_fla",
    "pe_ori",
    "dbe_acc",
    "dbe_com",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-shot PromptDA iBims inference + official eval",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-path", required=True, help="Local checkpoint path or Hugging Face model id")
    parser.add_argument("--encoder", choices=["vits", "vitb", "vitl", "vitg"], default="vitl")
    parser.add_argument("--ibims-root", default="data/ibims1", help="iBims dataset root")
    parser.add_argument(
        "--levels",
        nargs="+",
        choices=ALL_LEVELS,
        default=list(ALL_LEVELS),
        help="Difficulty levels to process",
    )
    parser.add_argument(
        "--output-root",
        default=str(SCRIPT_DIR / "output"),
        help="Base output directory used when --run-dir is not set",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Existing or new run directory; required for --skip-infer if predictions are not under --output-root",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=1008,
        help="Maximum RGB side length before rounding to a multiple of 14",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Maximum samples per level; 0 means all samples",
    )
    parser.add_argument(
        "--clamp-prediction",
        action="store_true",
        help="Clamp finite positive predictions to manifest depth ranges before saving",
    )
    parser.add_argument("--skip-infer", action="store_true", help="Skip inference and use existing predictions")
    parser.add_argument("--skip-eval", action="store_true", help="Skip official evaluation")
    return parser.parse_args()


def build_run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir:
        return Path(args.run_dir).expanduser().resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = sanitize_model_name(args.model_path)
    return Path(args.output_root).expanduser().resolve() / f"ibims_{model_name}_{timestamp}"


def manifest_for_level(ibims_root: Path, level: str) -> Path:
    return ibims_root / SYNTHETIC_RAW_DIR_NAME / "manifests" / f"ibims_{level}.jsonl"


def run_inference(args: argparse.Namespace, run_dir: Path) -> None:
    ibims_root = resolve_root(args.ibims_root)
    model = load_model(args.model_path, args.encoder)

    for level in args.levels:
        manifest_path = manifest_for_level(ibims_root, level)
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Missing synthetic iBims manifest: {manifest_path}. "
                "Generate or provide synthetic raw depth before running this pipeline."
            )

        pred_dir = run_dir / "predictions" / level
        summary = infer_manifest(
            manifest_path,
            pred_dir,
            model,
            args.input_size,
            args.max_samples,
            args.clamp_prediction,
        )
        write_infer_args(pred_dir, args, {"level": level, **summary})
        print(f"[infer] {level}: wrote {summary['num_predictions']} predictions to {pred_dir}")


def parse_eval_stdout(text: str) -> Dict[str, float]:
    results: Dict[str, float] = {}
    in_block = False
    for line in text.splitlines():
        stripped = line.strip()
        if not in_block:
            if stripped == "Results:":
                in_block = True
            continue
        if not stripped:
            continue
        match = re.match(r"(\S+)\s*=\s*([\d.eE+\-]+)", stripped)
        if match:
            results[match.group(1)] = float(match.group(2))
        else:
            break
    return results


def run_evaluation(args: argparse.Namespace, run_dir: Path) -> Dict[str, Dict[str, float]]:
    ibims_root = resolve_root(args.ibims_root)
    all_metrics: Dict[str, Dict[str, float]] = {}

    for level in args.levels:
        pred_dir = run_dir / "predictions" / level
        if not pred_dir.is_dir():
            raise FileNotFoundError(f"Missing prediction directory for {level}: {pred_dir}")

        workspace = run_dir / "official_eval" / level / "workspace"
        log_path = workspace.parent / "official_eval_stdout.txt"
        print(f"[eval] {level}: preparing workspace {workspace}")
        eval_script, names = prepare_workspace(ibims_root, pred_dir, workspace, args.max_samples)

        print(f"[eval] {level}: running official eval on {len(names)} samples")
        stdout = run_official_eval(eval_script, workspace, log_path)
        metrics = parse_eval_stdout(stdout)
        all_metrics[level] = metrics

        if metrics:
            print(f"[eval] {level}: extracted {len(metrics)} metrics")
        else:
            print(f"[eval] {level}: WARNING - no metrics parsed from official output")

    return all_metrics


def write_summary(run_dir: Path, all_metrics: Dict[str, Dict[str, float]]) -> None:
    if not all_metrics:
        print("[eval] No metrics collected.")
        return

    extra_keys: List[str] = []
    for metrics in all_metrics.values():
        for key in metrics:
            if key not in RESULT_METRIC_KEYS and key not in extra_keys:
                extra_keys.append(key)
    metric_keys = RESULT_METRIC_KEYS + extra_keys

    summary_path = run_dir / "eval_summary.csv"
    with open(summary_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["level"] + metric_keys)
        writer.writeheader()
        for level in [item for item in ALL_LEVELS if item in all_metrics]:
            writer.writerow({"level": level, **{key: all_metrics[level].get(key) for key in metric_keys}})

    print(f"Eval summary saved to: {summary_path}")
    print_metrics_table(all_metrics)


def print_metrics_table(all_metrics: Dict[str, Dict[str, float]]) -> None:
    levels = [level for level in ALL_LEVELS if level in all_metrics]
    keys: List[str] = []
    for key in RESULT_METRIC_KEYS:
        if any(key in metrics for metrics in all_metrics.values()):
            keys.append(key)
    for metrics in all_metrics.values():
        for key in metrics:
            if key not in keys:
                keys.append(key)

    if not keys:
        return

    col_width = 10
    header = f"{'metric':<12}" + "".join(f"{level:>{col_width}}" for level in levels)
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for key in keys:
        row = f"{key:<12}"
        for level in levels:
            value = all_metrics[level].get(key)
            row += f"{value:{col_width}.4f}" if value is not None else f"{'-':>{col_width}}"
        print(row)
    print(sep)


def write_run_args(run_dir: Path, args: argparse.Namespace) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "run_args.json", "w", encoding="utf-8") as file:
        json.dump(vars(args), file, indent=2, sort_keys=True)


def main() -> None:
    args = parse_args()
    validate_common_args(args, require_model_path=not args.skip_infer)

    run_dir = build_run_dir(args)
    if args.skip_infer and not run_dir.is_dir():
        raise FileNotFoundError(f"--run-dir does not exist (needed when --skip-infer): {run_dir}")

    run_dir.mkdir(parents=True, exist_ok=True)
    write_run_args(run_dir, args)
    print(f"Run directory: {run_dir}")

    if not args.skip_infer:
        run_inference(args, run_dir)

    if not args.skip_eval:
        all_metrics = run_evaluation(args, run_dir)
        write_summary(run_dir, all_metrics)


if __name__ == "__main__":
    main()
