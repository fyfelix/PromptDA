#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from common import (
    DEFAULT_DATASET_PATH,
    DEFAULT_DEPTH_SCALE,
    DEFAULT_MAX_DEPTH,
    DEFAULT_MAX_SIZE,
    DEFAULT_MODEL_PATH,
    DEFAULT_RAW_TYPE,
    PROJECT_ROOT,
    default_output_path,
    display_path,
    preprocess_for_promptda,
    project_path,
    read_depth_meters,
    read_depth_shape,
    read_rgb,
    validate_static_inputs,
    write_json,
)


if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PromptDA inference on HAMMER jsonl data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH), help="Local PromptDA checkpoint path")
    parser.add_argument("--encoder", default="vitl", choices=["vits", "vitb", "vitl", "vitg"], help="PromptDA encoder")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH), help="HAMMER jsonl path")
    parser.add_argument("--raw-type", default=DEFAULT_RAW_TYPE, choices=["d435", "l515", "tof"], help="Raw depth sensor field")
    parser.add_argument("--output", default=None, help="Prediction output directory")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "mps"], help="Inference device")
    parser.add_argument("--depth-scale", type=float, default=DEFAULT_DEPTH_SCALE, help="Depth PNG scale factor")
    parser.add_argument("--max-depth", type=float, default=DEFAULT_MAX_DEPTH, help="Maximum valid raw depth in meters")
    parser.add_argument("--max-size", type=int, default=DEFAULT_MAX_SIZE, help="Maximum model input side before 14x resizing")
    parser.add_argument("--batch-size", type=int, default=1, help="Inference batch size for same-shaped model inputs")
    parser.add_argument("--sample-check-limit", type=int, default=16, help="Number of samples to file-check during dry-run")
    parser.add_argument("--limit", type=int, default=0, help="Optional sample limit for debugging; 0 means all samples")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without loading the model or writing predictions")
    return parser.parse_args()


def resolve_device(name: str):
    import torch

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


def run_dry_run(args: argparse.Namespace, output_path: Path) -> int:
    samples = validate_static_inputs(
        args.model_path,
        args.dataset,
        args.raw_type,
        output_path,
        sample_check_limit=args.sample_check_limit,
        require_model=True,
    )
    print("Dry-run OK")
    print(f"  model:   {display_path(project_path(args.model_path))}")
    print(f"  dataset: {display_path(project_path(args.dataset))}")
    print(f"  raw:     {args.raw_type}")
    print(f"  output:  {display_path(output_path)}")
    print(f"  samples: {len(samples)}")
    print(f"  checked: {len(samples) if args.sample_check_limit <= 0 else min(args.sample_check_limit, len(samples))}")
    return 0


def flush_batch(
    model,
    batch_samples,
    batch_images: list[np.ndarray],
    batch_depths: list[np.ndarray],
    batch_output_shapes: list[tuple[int, int]],
    device,
    output_path: Path,
) -> None:
    import torch

    if not batch_samples:
        return

    image_tensor = torch.from_numpy(np.stack(batch_images)).permute(0, 3, 1, 2).to(device)
    depth_tensor = torch.from_numpy(np.stack(batch_depths)).unsqueeze(1).to(device)
    depth_pred = model.predict(image_tensor, depth_tensor).detach().float().cpu().numpy().astype(np.float32)

    for index, sample in enumerate(batch_samples):
        pred = depth_pred[index]
        if pred.ndim == 3:
            pred = pred[0]
        target_h, target_w = batch_output_shapes[index]
        if pred.shape != (target_h, target_w):
            pred = cv2.resize(pred, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        np.save(output_path / f"{sample.name}.npy", pred.astype(np.float32, copy=False))


def main() -> int:
    args = parse_args()
    model_path = project_path(args.model_path)
    dataset_path = project_path(args.dataset)
    output_path = project_path(args.output) if args.output else default_output_path(model_path, args.raw_type)

    if args.dry_run:
        return run_dry_run(args, output_path)

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.max_size < 14:
        raise ValueError("--max-size must be >= 14")

    samples = validate_static_inputs(
        model_path,
        dataset_path,
        args.raw_type,
        output_path,
        sample_check_limit=args.sample_check_limit,
        require_model=True,
    )
    if args.limit > 0:
        samples = samples[: args.limit]

    output_path.mkdir(parents=True, exist_ok=True)
    write_json(
        output_path / "args.json",
        {
            **vars(args),
            "model_path": str(model_path),
            "dataset": str(dataset_path),
            "output": str(output_path),
            "resolved_samples": len(samples),
            "prediction_layout": "flat",
        },
    )

    import torch
    from promptda.promptda import PromptDA

    device = resolve_device(args.device)
    print(f"Loading PromptDA model: {display_path(model_path)}")
    model = PromptDA.from_pretrained(
        str(model_path),
        model_kwargs={"encoder": args.encoder},
    ).to(device).eval()
    print(
        f"Running HAMMER inference on {len(samples)} samples with "
        f"device={device}, batch_size={args.batch_size}, max_size={args.max_size}"
    )

    start_time = time.time()
    batch_samples = []
    batch_images: list[np.ndarray] = []
    batch_depths: list[np.ndarray] = []
    batch_output_shapes: list[tuple[int, int]] = []
    pending_model_shape: Optional[tuple[int, int]] = None

    with torch.inference_mode():
        for offset, sample in enumerate(samples, start=1):
            rgb = read_rgb(sample.rgb_path)
            prompt_depth = read_depth_meters(sample.raw_depth_path, args.depth_scale, args.max_depth)
            image_input, depth_input = preprocess_for_promptda(
                rgb,
                prompt_depth,
                sample.name,
                max_size=args.max_size,
            )
            model_shape = image_input.shape[:2]
            output_shape = read_depth_shape(sample.gt_depth_path)

            if pending_model_shape is not None and model_shape != pending_model_shape:
                flush_batch(
                    model,
                    batch_samples,
                    batch_images,
                    batch_depths,
                    batch_output_shapes,
                    device,
                    output_path,
                )
                batch_samples.clear()
                batch_images.clear()
                batch_depths.clear()
                batch_output_shapes.clear()
                pending_model_shape = None

            pending_model_shape = model_shape
            batch_samples.append(sample)
            batch_images.append(image_input)
            batch_depths.append(depth_input)
            batch_output_shapes.append(output_shape)

            if len(batch_samples) >= args.batch_size:
                flush_batch(
                    model,
                    batch_samples,
                    batch_images,
                    batch_depths,
                    batch_output_shapes,
                    device,
                    output_path,
                )
                batch_samples.clear()
                batch_images.clear()
                batch_depths.clear()
                batch_output_shapes.clear()
                pending_model_shape = None

            if offset == 1 or offset % 50 == 0 or offset == len(samples):
                elapsed = time.time() - start_time
                fps = offset / elapsed if elapsed > 0 else 0.0
                print(f"[{offset}/{len(samples)}] {sample.name} ({fps:.2f} FPS)")

        flush_batch(
            model,
            batch_samples,
            batch_images,
            batch_depths,
            batch_output_shapes,
            device,
            output_path,
        )

    print(f"Inference complete: {display_path(output_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
