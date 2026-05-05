#!/usr/bin/env python3
"""Run PromptDA inference for iBims and save official *_results.mat files."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


IBIMS_DEPTH_MAX_M = 50.0
IBIMS_DEPTH_SCALE = 65535.0 / IBIMS_DEPTH_MAX_M
SYNTHETIC_RAW_DIR_NAME = "ibims1_synthetic_raw_depth"
EXPECTED_SHAPE = (480, 640)


def get_device() -> str:
    import torch

    return (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in ("yes", "true", "t", "y", "1"):
        return True
    if value in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PromptDA inference for iBims official MAT evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", required=True, help="iBims synthetic JSONL manifest")
    parser.add_argument(
        "--model-path",
        required=True,
        help="Local checkpoint path or Hugging Face model id",
    )
    parser.add_argument(
        "--encoder",
        choices=["vits", "vitb", "vitl", "vitg"],
        default="vitl",
        help="PromptDA DINOv2 encoder",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Prediction directory; defaults under evaluation_ibims/output",
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
        help="Maximum number of manifest rows to run; 0 means all samples",
    )
    parser.add_argument(
        "--clamp-prediction",
        type=str2bool,
        default=False,
        help="Clamp finite positive predictions to the manifest depth range before saving",
    )
    return parser.parse_args()


def looks_like_hf_model_id(model_path: str) -> bool:
    path = Path(os.path.expanduser(model_path))
    if path.exists():
        return False
    if model_path.startswith((".", "/", "~")):
        return False
    return "/" in model_path


def validate_common_args(args: argparse.Namespace, require_model_path: bool = True) -> None:
    if args.max_samples < 0:
        raise ValueError("--max-samples must be >= 0")
    if args.input_size < 0:
        raise ValueError("--input-size must be >= 0")

    args.model_path = os.path.expanduser(args.model_path)
    if (
        require_model_path
        and not os.path.exists(args.model_path)
        and not looks_like_hf_model_id(args.model_path)
    ):
        raise FileNotFoundError(
            f"Model path does not exist and does not look like a Hugging Face id: {args.model_path}"
        )


def resolve_path(base: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return (base / path).resolve()


def load_manifest(manifest_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    required_keys = ("sample_id", "rgb", "raw_depth", "depth", "depth-range", "depth_scale")

    with open(manifest_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            if row.get("dataset", "ibims") != "ibims":
                raise ValueError(f"{manifest_path}:{line_number} is not an iBims row")

            for key in required_keys:
                if key not in row:
                    raise ValueError(f"{manifest_path}:{line_number} missing required key: {key}")
            rows.append(row)

    if not rows:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    return rows


def infer_difficulty(manifest_path: Path, rows: List[Dict[str, Any]]) -> str:
    difficulty = rows[0].get("difficulty")
    if difficulty:
        return str(difficulty)
    stem = manifest_path.stem
    return stem[len("ibims_") :] if stem.startswith("ibims_") else stem


def sanitize_model_name(model_path: str) -> str:
    path = Path(model_path)
    if path.exists():
        name = path.stem
    else:
        name = str(model_path).rstrip("/").split("/")[-1]
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in name)


def default_output_dir(manifest_path: Path, rows: List[Dict[str, Any]], model_path: str) -> Path:
    difficulty = infer_difficulty(manifest_path, rows)
    model_name = sanitize_model_name(model_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return SCRIPT_DIR / "output" / f"ibims_{model_name}_{timestamp}" / "predictions" / difficulty


def load_model(model_path: str, encoder: str):
    from promptda.promptda import PromptDA

    device = get_device()
    print(f"Loading PromptDA model: encoder={encoder}, model_path={model_path}, device={device}")
    model = PromptDA.from_pretrained(
        model_path,
        model_kwargs={"encoder": encoder},
    )
    return model.to(device).eval()


def ensure_multiple_of(value: float, multiple: int = 14) -> int:
    return max(multiple, int(value // multiple * multiple))


def load_rgb_tensor(rgb_path: Path, max_size: int, multiple_of: int = 14):
    import cv2
    import numpy as np
    import torch

    image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not load RGB image: {rgb_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0

    height, width = image.shape[:2]
    scale = 1.0
    if max_size > 0 and max(height, width) > max_size:
        scale = max_size / max(height, width)

    target_h = ensure_multiple_of(height * scale, multiple_of)
    target_w = ensure_multiple_of(width * scale, multiple_of)
    if (target_h, target_w) != (height, width):
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        image = cv2.resize(image, (target_w, target_h), interpolation=interpolation)

    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    return tensor, image


def load_depth_array(depth_path: Path, depth_scale: float, max_depth: float):
    import cv2
    import numpy as np

    if str(depth_path).endswith(".npz"):
        depth = np.load(depth_path)["depth"]
    else:
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise ValueError(f"Could not load depth image: {depth_path}")

    depth = np.asarray(depth).astype(np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    if not str(depth_path).endswith(".npz"):
        depth = depth / depth_scale

    depth[~np.isfinite(depth)] = 0.0
    depth[depth < 0] = 0.0
    if max_depth > 0:
        depth[depth > max_depth] = 0.0
    return depth


def depth_to_tensor(depth):
    import torch

    return torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)


def load_gt_shape(gt_depth_path: Path) -> Tuple[int, int]:
    import cv2

    gt_depth = cv2.imread(str(gt_depth_path), cv2.IMREAD_UNCHANGED)
    if gt_depth is None:
        raise ValueError(f"Could not load GT depth image: {gt_depth_path}")
    return gt_depth.shape[:2]


def row_depth_range(row: Dict[str, Any]) -> Tuple[float, float]:
    depth_range = row["depth-range"]
    if not isinstance(depth_range, list) or len(depth_range) != 2:
        raise ValueError(f"Invalid depth-range for {row.get('sample_id')}: {depth_range}")
    return float(depth_range[0]), float(depth_range[1])


def predict_row(
    model: Any,
    row: Dict[str, Any],
    manifest_dir: Path,
    input_size: int,
    clamp_prediction: bool,
):
    import numpy as np
    import torch
    import torch.nn.functional as F

    rgb_path = resolve_path(manifest_dir, row["rgb"])
    raw_depth_path = resolve_path(manifest_dir, row["raw_depth"])
    gt_depth_path = resolve_path(manifest_dir, row["depth"])
    min_depth, max_depth = row_depth_range(row)
    depth_scale = float(row["depth_scale"])

    rgb_tensor, _ = load_rgb_tensor(rgb_path, input_size)
    prompt_depth = load_depth_array(raw_depth_path, depth_scale, max_depth)
    prompt_tensor = depth_to_tensor(prompt_depth)
    target_h, target_w = load_gt_shape(gt_depth_path)

    device = get_device()
    rgb_tensor = rgb_tensor.to(device)
    prompt_tensor = prompt_tensor.to(device)

    with torch.no_grad():
        pred = model.predict(rgb_tensor, prompt_tensor)
    if pred.ndim == 3:
        pred = pred.unsqueeze(1)
    if pred.ndim != 4:
        raise ValueError(f"Unexpected PromptDA output shape: {tuple(pred.shape)}")

    pred = F.interpolate(
        pred,
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    pred_np = pred.detach().cpu().numpy().astype(np.float32)

    if clamp_prediction:
        finite_positive = np.isfinite(pred_np) & (pred_np > 0)
        pred_np[finite_positive] = np.clip(pred_np[finite_positive], min_depth, max_depth)

    official_depth = np.full(pred_np.shape, np.nan, dtype=np.float32)
    valid_pred = np.isfinite(pred_np) & (pred_np > 0)
    official_depth[valid_pred] = pred_np[valid_pred]
    return official_depth


def limited_rows(rows: List[Dict[str, Any]], max_samples: int) -> List[Dict[str, Any]]:
    if max_samples <= 0:
        return rows
    return rows[:max_samples]


def infer_manifest(
    manifest_path: Path,
    output_dir: Path,
    model: Any,
    input_size: int,
    max_samples: int,
    clamp_prediction: bool,
) -> Dict[str, Any]:
    import numpy as np
    from scipy.io import savemat
    from tqdm import tqdm

    rows = limited_rows(load_manifest(manifest_path), max_samples)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: List[str] = []
    for row in tqdm(rows, desc=f"Infer {infer_difficulty(manifest_path, rows)}"):
        sample_id = str(row["sample_id"])
        depth = predict_row(model, row, manifest_path.parent, input_size, clamp_prediction)
        if depth.shape != EXPECTED_SHAPE:
            raise ValueError(f"{sample_id}: expected prediction shape {EXPECTED_SHAPE}, got {depth.shape}")

        prediction_path = output_dir / f"{sample_id}_results.mat"
        savemat(prediction_path, {"pred_depths": depth.astype(np.float32)})
        written.append(str(prediction_path))

    return {
        "manifest": str(manifest_path),
        "output_dir": str(output_dir),
        "num_predictions": len(written),
    }


def json_ready_args(args: argparse.Namespace) -> Dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def write_infer_args(output_dir: Path, args: argparse.Namespace, extra: Dict[str, Any]) -> None:
    with open(output_dir / "infer_args.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                **json_ready_args(args),
                **extra,
                "device": get_device(),
                "resolved_model_module": "promptda.promptda",
                "resolved_model_class": "PromptDA",
                "prediction_kind": "metric_depth_meters",
                "uses_prompt_depth": True,
            },
            file,
            indent=2,
            sort_keys=True,
        )


def main() -> None:
    args = parse_args()
    validate_common_args(args)

    manifest_path = Path(args.manifest).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")

    rows = load_manifest(manifest_path)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else default_output_dir(manifest_path, rows, args.model_path).resolve()
    )

    model = load_model(args.model_path, args.encoder)
    summary = infer_manifest(
        manifest_path,
        output_dir,
        model,
        args.input_size,
        args.max_samples,
        args.clamp_prediction,
    )
    write_infer_args(output_dir, args, summary)
    print(f"Wrote {summary['num_predictions']} official iBims predictions to: {output_dir}")


if __name__ == "__main__":
    main()
