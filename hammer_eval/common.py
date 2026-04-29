from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Union

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "ckpts" / "promptda_vitl.ckpt"
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "HAMMER" / "test.jsonl"
DEFAULT_RAW_TYPE = "d435"
DEFAULT_DEPTH_SCALE = 1000.0
DEFAULT_MAX_DEPTH = 6.0
DEFAULT_MAX_SIZE = 1008
PATCH_SIZE = 14

RAW_DEPTH_FIELDS = {
    "d435": "d435_depth",
    "l515": "l515_depth",
    "tof": "tof_depth",
}


@dataclass(frozen=True)
class HammerSample:
    index: int
    item: dict
    name: str
    rgb_path: Path
    raw_depth_path: Path
    gt_depth_path: Path
    depth_range: tuple[float, float]


def project_path(path: Union[str, Path]) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def display_path(path: Union[str, Path]) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def default_output_path(model_path: Union[str, Path], raw_type: str) -> Path:
    model_stem = Path(model_path).stem
    return PROJECT_ROOT / "hammer_eval" / "outputs" / f"hammer_{model_stem}_data_{raw_type}"


def read_jsonl(path: Union[str, Path]) -> list[dict]:
    records = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {display_path(path)}:{line_no}: {exc}") from exc
    if not records:
        raise ValueError(f"HAMMER jsonl is empty: {display_path(path)}")
    return records


def prediction_name(item: dict, rgb_path: Union[str, Path]) -> str:
    parts = Path(rgb_path).parts
    if len(parts) >= 4:
        scene_name = parts[-4]
    else:
        scene_name = str(item.get("seq_name") or "unknown_scene")
    return f"{scene_name}#{Path(rgb_path).stem}"


def _require_fields(item: dict, fields: Iterable[str], dataset_path: Path, index: int) -> None:
    missing = [field for field in fields if field not in item]
    if missing:
        raise ValueError(
            f"Sample {index} in {display_path(dataset_path)} is missing fields: {missing}"
        )


def load_hammer_samples(dataset_path: Union[str, Path], raw_type: str) -> list[HammerSample]:
    dataset_path = project_path(dataset_path)
    raw_type = raw_type.lower()
    if raw_type not in RAW_DEPTH_FIELDS:
        raise ValueError(f"Invalid raw_type '{raw_type}'. Expected one of {sorted(RAW_DEPTH_FIELDS)}")

    records = read_jsonl(dataset_path)
    root = dataset_path.parent
    raw_depth_field = RAW_DEPTH_FIELDS[raw_type]
    samples = []

    for index, item in enumerate(records):
        _require_fields(item, ("rgb", "depth", raw_depth_field, "depth-range"), dataset_path, index)
        depth_range = item["depth-range"]
        if not isinstance(depth_range, (list, tuple)) or len(depth_range) != 2:
            raise ValueError(
                f"Sample {index} in {display_path(dataset_path)} has invalid depth-range: {depth_range}"
            )

        rgb_path = root / item["rgb"]
        raw_depth_path = root / item[raw_depth_field]
        gt_depth_path = root / item["depth"]
        samples.append(
            HammerSample(
                index=index,
                item=item,
                name=prediction_name(item, rgb_path),
                rgb_path=rgb_path,
                raw_depth_path=raw_depth_path,
                gt_depth_path=gt_depth_path,
                depth_range=(float(depth_range[0]), float(depth_range[1])),
            )
        )

    return samples


def dataset_depth_range(samples: list[HammerSample]) -> tuple[float, float]:
    if not samples:
        raise ValueError("Cannot read depth range from an empty HAMMER sample list")
    return samples[0].depth_range


def check_sample_files(samples: list[HammerSample], limit: int) -> None:
    checked = samples if limit <= 0 else samples[:limit]
    missing = []
    for sample in checked:
        for label, path in (
            ("rgb", sample.rgb_path),
            ("raw_depth", sample.raw_depth_path),
            ("gt_depth", sample.gt_depth_path),
        ):
            if not path.is_file():
                missing.append(f"sample {sample.index} {label}: {display_path(path)}")
    if missing:
        preview = "\n".join(f"  - {item}" for item in missing[:20])
        suffix = "" if len(missing) <= 20 else f"\n  ... and {len(missing) - 20} more"
        raise FileNotFoundError(f"Missing HAMMER files:\n{preview}{suffix}")


def validate_static_inputs(
    model_path: Union[str, Path],
    dataset_path: Union[str, Path],
    raw_type: str,
    output_path: Union[str, Path],
    sample_check_limit: int = 16,
    require_model: bool = True,
) -> list[HammerSample]:
    model_path = project_path(model_path)
    dataset_path = project_path(dataset_path)
    output_path = project_path(output_path)

    if require_model and not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint does not exist: {display_path(model_path)}")
    if not dataset_path.is_file():
        raise FileNotFoundError(f"HAMMER jsonl does not exist: {display_path(dataset_path)}")

    samples = load_hammer_samples(dataset_path, raw_type)
    check_sample_files(samples, sample_check_limit)

    parent = output_path.parent
    if parent.exists() and not parent.is_dir():
        raise NotADirectoryError(f"Output parent is not a directory: {display_path(parent)}")

    return samples


def read_rgb(path: Union[str, Path]) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Failed to read RGB image: {display_path(path)}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def read_depth_meters(
    path: Union[str, Path],
    depth_scale: float = DEFAULT_DEPTH_SCALE,
    max_depth: float = DEFAULT_MAX_DEPTH,
) -> np.ndarray:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise ValueError(f"Failed to read depth image: {display_path(path)}")
    depth = depth.astype(np.float32) / float(depth_scale)
    invalid = (~np.isfinite(depth)) | (depth <= 0.0) | (depth > float(max_depth))
    depth[invalid] = 0.0
    return depth


def read_gt_depth_and_mask(
    path: Union[str, Path],
    depth_scale: float,
    min_depth: float,
    max_depth: float,
) -> tuple[np.ndarray, np.ndarray]:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise ValueError(f"Failed to read GT depth image: {display_path(path)}")
    depth = depth.astype(np.float32) / float(depth_scale)
    valid_mask = np.isfinite(depth) & (depth >= float(min_depth)) & (depth <= float(max_depth))
    depth = depth.astype(np.float32, copy=True)
    depth[~valid_mask] = float(min_depth)
    return depth, valid_mask


def read_depth_shape(path: Union[str, Path]) -> tuple[int, int]:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise ValueError(f"Failed to read depth image: {display_path(path)}")
    return int(depth.shape[0]), int(depth.shape[1])


def constrain_model_size(height: int, width: int, max_size: int, multiple_of: int = PATCH_SIZE) -> tuple[int, int]:
    max_size = int(max_size) // multiple_of * multiple_of
    if max_size < multiple_of:
        raise ValueError(f"max_size must be at least {multiple_of}, got {max_size}")

    scale = min(1.0, float(max_size) / float(max(height, width)))
    target_h = int((height * scale) // multiple_of * multiple_of)
    target_w = int((width * scale) // multiple_of * multiple_of)
    target_h = max(multiple_of, target_h)
    target_w = max(multiple_of, target_w)
    return target_h, target_w


def ensure_prompt_depth_valid(depth: np.ndarray, sample_name: str) -> None:
    valid = np.isfinite(depth) & (depth > 0.0)
    if not valid.any():
        raise ValueError(f"Raw prompt depth has no valid pixels for {sample_name}")

    depth_min = float(np.nanmin(depth))
    depth_max = float(np.nanmax(depth))
    if depth_max <= depth_min:
        raise ValueError(
            f"Raw prompt depth has zero dynamic range for {sample_name}: min={depth_min}, max={depth_max}"
        )


def preprocess_for_promptda(
    rgb: np.ndarray,
    prompt_depth: np.ndarray,
    sample_name: str,
    max_size: int = DEFAULT_MAX_SIZE,
    multiple_of: int = PATCH_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    ensure_prompt_depth_valid(prompt_depth, sample_name)
    target_h, target_w = constrain_model_size(rgb.shape[0], rgb.shape[1], max_size, multiple_of)

    rgb_float = rgb.astype(np.float32) / 255.0
    prompt_float = prompt_depth.astype(np.float32, copy=False)

    if rgb.shape[:2] != (target_h, target_w):
        rgb_float = cv2.resize(rgb_float, (target_w, target_h), interpolation=cv2.INTER_AREA)
    if prompt_depth.shape[:2] != (target_h, target_w):
        prompt_float = cv2.resize(prompt_float, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    rgb_float = np.ascontiguousarray(rgb_float.astype(np.float32, copy=False))
    prompt_float = np.ascontiguousarray(prompt_float.astype(np.float32, copy=False))
    ensure_prompt_depth_valid(prompt_float, sample_name)
    return rgb_float, prompt_float


def write_json(path: Union[str, Path], payload: dict) -> None:
    serializable = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in payload.items()
    }
    with Path(path).open("w", encoding="utf-8") as file:
        json.dump(serializable, file, indent=2, ensure_ascii=False)
        file.write("\n")
