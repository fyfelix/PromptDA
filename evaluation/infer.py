#!/usr/bin/env python3

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

DEFAULT_INTRINSICS_PATH = "data/TRansPose/sequences/intrinsics.txt"


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("yes", "true", "t", "y", "1"):
        return True
    if value in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "PromptDA inference for HAMMER, ClearPose, DREDS, or TRansPose evaluation"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--encoder",
        type=str,
        choices=["vits", "vitb", "vitl", "vitg"],
        default="vitl",
        help="PromptDA DINOv2 encoder",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Local checkpoint path or Hugging Face model id",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="HAMMER, ClearPose, DREDS, or TRansPose JSONL path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output_dir",
        help="Directory for run metadata, predictions/, and visualizations/",
    )
    parser.add_argument(
        "--prediction-dir",
        type=str,
        default=None,
        help="Directory for per-sample .npy predictions",
    )
    parser.add_argument(
        "--visualization-dir",
        type=str,
        default=None,
        help="Directory for optional RGB/depth visualization images",
    )
    parser.add_argument(
        "--raw-type",
        type=str,
        required=True,
        choices=["d435", "l515", "tof"],
        help="Raw depth field used as PromptDA prompt depth; ClearPose only supports d435",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=1008,
        help="Maximum RGB side length before rounding to a multiple of 14",
    )
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=1000.0,
        help="Scale for uint depth PNG values",
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=6.0,
        help="Fallback maximum raw prompt depth in meters",
    )
    parser.add_argument(
        "--image-min",
        type=float,
        default=0.1,
        help="Visualization minimum depth",
    )
    parser.add_argument(
        "--image-max",
        type=float,
        default=5.0,
        help="Visualization maximum depth",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Path batching only; PromptDA inference is run one sample at a time",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers for path loading",
    )
    parser.add_argument(
        "--save-vis", action="store_true", help="Save optional RGB/depth visualizations"
    )
    parser.add_argument(
        "--intrinsics-path",
        type=str,
        default=DEFAULT_INTRINSICS_PATH,
        help="Camera intrinsics text file for TRansPose point cloud visualizations",
    )
    parser.add_argument(
        "--pc-rot-x-deg",
        type=float,
        default=25.0,
        help="TRansPose point cloud view rotation around X axis in degrees",
    )
    parser.add_argument(
        "--pc-rot-y-deg",
        type=float,
        default=15.0,
        help="TRansPose point cloud view rotation around Y axis in degrees",
    )
    parser.add_argument(
        "--pc-knn-k",
        type=int,
        default=16,
        help="KNN neighbors for TRansPose predicted point cloud floater filtering",
    )
    parser.add_argument(
        "--pc-knn-std-ratio",
        type=float,
        default=2.0,
        help="Mean-distance std ratio threshold for TRansPose point cloud filtering",
    )
    parser.add_argument(
        "--disable-pc-knn-filter",
        action="store_true",
        help="Disable KNN filtering for TRansPose predicted point cloud visualization",
    )
    parser.add_argument(
        "--clamp-prediction",
        type=str2bool,
        default=False,
        help="Clamp saved predictions to the dataset depth range",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Maximum number of dataset samples to run; 0 means all samples",
    )
    return parser.parse_args()


if __name__ == "__main__" and any(arg in ("-h", "--help") for arg in sys.argv[1:]):
    parse_arguments()
    sys.exit(0)


import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset import (
    limit_dataset_for_eval,
    load_test_dataset,
    sample_name_for_sample,
)
from promptda.promptda import PromptDA


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


def looks_like_hf_model_id(model_path):
    path = Path(os.path.expanduser(model_path))
    if path.exists():
        return False
    if model_path.startswith((".", "/", "~")):
        return False
    return "/" in model_path


def validate_inputs(args):
    if not os.path.exists(args.dataset):
        print(f"Error: dataset file '{args.dataset}' does not exist")
        sys.exit(1)
    if args.max_samples < 0:
        print("Error: --max-samples must be >= 0")
        sys.exit(1)

    args.model_path = os.path.expanduser(args.model_path)
    if not os.path.exists(args.model_path) and not looks_like_hf_model_id(args.model_path):
        print(
            f"Error: model path '{args.model_path}' does not exist. "
            "Pass a local checkpoint path or a Hugging Face repo id."
        )
        sys.exit(1)

    if args.prediction_dir is None:
        args.prediction_dir = str(Path(args.output) / "predictions")
    if args.visualization_dir is None:
        args.visualization_dir = str(Path(args.output) / "visualizations")

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.prediction_dir, exist_ok=True)
    if args.save_vis:
        os.makedirs(args.visualization_dir, exist_ok=True)


def load_model(args):
    print(
        "Loading PromptDA model: "
        f"encoder={args.encoder}, model_path={args.model_path}, device={DEVICE}"
    )
    model = PromptDA.from_pretrained(
        args.model_path,
        model_kwargs={"encoder": args.encoder},
    )
    model = model.to(DEVICE).eval()
    return model


def ensure_multiple_of(value, multiple=14):
    return max(multiple, int(value // multiple * multiple))


def load_rgb_tensor(rgb_path, max_size, multiple_of=14):
    image = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not load RGB image: {rgb_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_shape = image.shape[:2]
    image = image.astype(np.float32) / 255.0

    h, w = image.shape[:2]
    scale = 1.0
    if max_size > 0 and max(h, w) > max_size:
        scale = max_size / max(h, w)

    target_h = ensure_multiple_of(h * scale, multiple_of)
    target_w = ensure_multiple_of(w * scale, multiple_of)
    if (target_h, target_w) != (h, w):
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        image = cv2.resize(image, (target_w, target_h), interpolation=interpolation)

    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    return tensor, original_shape, image


def load_depth_array(depth_path, depth_scale, max_depth):
    if depth_path.endswith(".npz"):
        depth = np.load(depth_path)["depth"]
    else:
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise ValueError(f"Could not load depth image: {depth_path}")

    depth = np.asarray(depth).astype(np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    if not depth_path.endswith(".npz"):
        depth = depth / depth_scale

    depth[~np.isfinite(depth)] = 0.0
    depth[depth < 0] = 0.0
    if max_depth > 0:
        depth[depth > max_depth] = 0.0
    return depth


def depth_to_tensor(depth):
    return torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)


def load_gt_shape(gt_depth_path):
    gt_depth = cv2.imread(gt_depth_path, cv2.IMREAD_UNCHANGED)
    if gt_depth is None:
        raise ValueError(f"Could not load GT depth from {gt_depth_path}")
    return gt_depth.shape[:2]


@torch.no_grad()
def predict_depth(model, rgb_path, raw_depth_path, gt_depth_path, args):
    rgb_tensor, original_shape, rgb_for_vis = load_rgb_tensor(rgb_path, args.input_size)
    prompt_depth = load_depth_array(raw_depth_path, args.depth_scale, args.max_depth)
    prompt_tensor = depth_to_tensor(prompt_depth)
    target_h, target_w = load_gt_shape(gt_depth_path)

    rgb_tensor = rgb_tensor.to(DEVICE)
    prompt_tensor = prompt_tensor.to(DEVICE)

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
    pred = pred.detach().cpu().numpy().astype(np.float32)

    if args.clamp_prediction:
        pred = np.clip(pred, args.min_depth, args.max_depth).astype(np.float32)

    return pred, rgb_for_vis, prompt_depth, original_shape


def colorize_depth(depth, image_min, image_max):
    depth = np.asarray(depth, dtype=np.float32)
    norm = (depth - image_min) / max(image_max - image_min, 1e-6)
    norm = np.clip(norm, 0.0, 1.0)
    colored = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    colored[depth <= 0] = 0
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def save_visualization(output_path, rgb, prompt_depth, pred_depth, args):
    pred_vis = colorize_depth(pred_depth, args.image_min, args.image_max)
    prompt_vis = colorize_depth(prompt_depth, args.image_min, args.image_max)
    prompt_vis = cv2.resize(
        prompt_vis,
        (rgb.shape[1], rgb.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    pred_vis = cv2.resize(
        pred_vis,
        (rgb.shape[1], rgb.shape[0]),
        interpolation=cv2.INTER_AREA,
    )
    rgb_u8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    grid = np.concatenate([rgb_u8, prompt_vis, pred_vis], axis=1)
    Image.fromarray(grid).save(output_path)


def load_intrinsics(path):
    intrinsics = np.loadtxt(path, dtype=np.float32)
    if intrinsics.shape != (3, 3):
        raise ValueError(
            f"Intrinsics matrix must have shape (3, 3), got {intrinsics.shape}"
        )
    return intrinsics


def scale_intrinsics(intrinsics, orig_hw, new_hw):
    sy = new_hw[0] / orig_hw[0]
    sx = new_hw[1] / orig_hw[1]
    scaled = intrinsics.copy()
    scaled[0, :] *= sx
    scaled[1, :] *= sy
    return scaled


def resize_to(image, target_hw, interpolation=cv2.INTER_LINEAR):
    target_h, target_w = target_hw
    if image.shape[:2] == (target_h, target_w):
        return image
    return cv2.resize(image, (target_w, target_h), interpolation=interpolation)


def filter_pointcloud_knn(points, colors, k=16, std_ratio=2.0):
    if k < 1 or points.shape[0] <= k:
        return points, colors

    try:
        from scipy.spatial import cKDTree

        neighbor_count = min(k + 1, points.shape[0])
        tree = cKDTree(points)
        distances, _ = tree.query(points, k=neighbor_count, workers=-1)
    except Exception:
        return points, colors

    if distances.ndim == 1:
        return points, colors

    mean_distances = distances[:, 1:].mean(axis=1)
    finite = np.isfinite(mean_distances)
    if not finite.any():
        return points, colors

    valid_mean_distances = mean_distances[finite]
    threshold = valid_mean_distances.mean() + std_ratio * valid_mean_distances.std()
    keep = finite & (mean_distances <= threshold)
    if not keep.any():
        return points, colors
    return points[keep], colors[keep]


def render_pointcloud_reproject(
    depth_map,
    intrinsics,
    rgb_img,
    rot_x_deg=25.0,
    rot_y_deg=15.0,
    bg_color=(255, 255, 255),
    knn_filter=True,
    knn_k=16,
    knn_std_ratio=2.0,
):
    depth_map = np.asarray(depth_map, dtype=np.float32).squeeze()
    height, width = depth_map.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    u, v = np.meshgrid(np.arange(width), np.arange(height))
    valid = (depth_map > 1e-8) & np.isfinite(depth_map)
    if not valid.any():
        return np.full((height, width, 3), bg_color, dtype=np.uint8)

    z = depth_map[valid]
    x = (u[valid] - cx) * z / fx
    y = (v[valid] - cy) * z / fy
    points = np.stack([x, y, z], axis=-1).astype(np.float32, copy=False)
    colors = np.clip(rgb_img, 0, 255).astype(np.uint8)[valid]
    if knn_filter:
        points, colors = filter_pointcloud_knn(
            points,
            colors,
            k=knn_k,
            std_ratio=knn_std_ratio,
        )

    center = points.mean(axis=0)
    points_centered = points - center

    rx = np.radians(rot_x_deg)
    ry = np.radians(rot_y_deg)
    cos_x, sin_x = np.cos(rx), np.sin(rx)
    cos_y, sin_y = np.cos(ry), np.sin(ry)

    x1 = points_centered[:, 0]
    y1 = points_centered[:, 1] * cos_x - points_centered[:, 2] * sin_x
    z1 = points_centered[:, 1] * sin_x + points_centered[:, 2] * cos_x
    x2 = x1 * cos_y + z1 * sin_y
    y2 = y1
    z2 = -x1 * sin_y + z1 * cos_y
    points_rot = np.stack([x2, y2, z2], axis=-1) + center
    z_new = points_rot[:, 2]
    keep = z_new > 1e-4
    if not keep.any():
        return np.full((height, width, 3), bg_color, dtype=np.uint8)

    u_proj = points_rot[keep, 0] * fx / z_new[keep] + cx
    v_proj = points_rot[keep, 1] * fy / z_new[keep] + cy
    z_buf = z_new[keep]
    c_buf = colors[keep]

    pad = int(max(height, width) * 0.3)
    canvas_h, canvas_w = height + 2 * pad, width + 2 * pad
    ui = np.round(u_proj + pad).astype(np.int32)
    vi = np.round(v_proj + pad).astype(np.int32)

    in_bounds = (ui >= 0) & (ui < canvas_w) & (vi >= 0) & (vi < canvas_h)
    ui = ui[in_bounds]
    vi = vi[in_bounds]
    z_buf = z_buf[in_bounds]
    c_buf = c_buf[in_bounds]
    if ui.size == 0:
        return np.full((height, width, 3), bg_color, dtype=np.uint8)

    order = np.argsort(-z_buf)
    ui = ui[order]
    vi = vi[order]
    c_buf = c_buf[order]

    canvas = np.full((canvas_h, canvas_w, 3), bg_color, dtype=np.uint8)
    canvas[vi, ui] = c_buf

    filled = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    filled[vi, ui] = 255
    kernel = np.ones((3, 3), dtype=np.uint8)
    filled_dilated = cv2.dilate(filled, kernel, iterations=1)
    holes = (filled_dilated > 0) & (filled == 0)
    if holes.any():
        for channel_idx in range(3):
            blurred = cv2.blur(canvas[:, :, channel_idx].astype(np.float32), (3, 3))
            canvas[:, :, channel_idx][holes] = blurred[holes].astype(np.uint8)

    rows = np.any(filled_dilated > 0, axis=1)
    cols = np.any(filled_dilated > 0, axis=0)
    if rows.any() and cols.any():
        row_min, row_max = np.where(rows)[0][[0, -1]]
        col_min, col_max = np.where(cols)[0][[0, -1]]
        margin = 10
        row_min = max(0, row_min - margin)
        row_max = min(canvas_h - 1, row_max + margin)
        col_min = max(0, col_min - margin)
        col_max = min(canvas_w - 1, col_max + margin)
        canvas = canvas[row_min : row_max + 1, col_min : col_max + 1]

    return resize_to(canvas, (height, width), interpolation=cv2.INTER_LINEAR)


def image_grid(images, rows, cols):
    if len(images) != rows * cols:
        raise ValueError(f"Expected {rows * cols} images, got {len(images)}")

    height, width = images[0].shape[:2]
    normalized = [
        resize_to(image, (height, width), interpolation=cv2.INTER_LINEAR).astype(
            np.uint8
        )
        for image in images
    ]
    row_images = [
        np.concatenate(normalized[row_idx * cols : (row_idx + 1) * cols], axis=1)
        for row_idx in range(rows)
    ]
    return np.concatenate(row_images, axis=0)


def create_transpose_visualization(
    rgb,
    raw_depth,
    pred_depth,
    gt_depth,
    intrinsics,
    args,
):
    rgb_u8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    target_hw = rgb_u8.shape[:2]
    raw_depth = resize_to(raw_depth, target_hw, interpolation=cv2.INTER_NEAREST)
    pred_depth = resize_to(pred_depth, target_hw, interpolation=cv2.INTER_LINEAR)
    gt_depth = resize_to(gt_depth, target_hw, interpolation=cv2.INTER_NEAREST)

    pred_pointcloud = render_pointcloud_reproject(
        pred_depth,
        intrinsics,
        rgb_u8,
        rot_x_deg=args.pc_rot_x_deg,
        rot_y_deg=args.pc_rot_y_deg,
        knn_filter=not args.disable_pc_knn_filter,
        knn_k=args.pc_knn_k,
        knn_std_ratio=args.pc_knn_std_ratio,
    )
    gt_pointcloud = render_pointcloud_reproject(
        gt_depth,
        intrinsics,
        rgb_u8,
        rot_x_deg=args.pc_rot_x_deg,
        rot_y_deg=args.pc_rot_y_deg,
        knn_filter=False,
    )

    return image_grid(
        [
            rgb_u8,
            colorize_depth(raw_depth, args.image_min, args.image_max),
            colorize_depth(pred_depth, args.image_min, args.image_max),
            colorize_depth(gt_depth, args.image_min, args.image_max),
            pred_pointcloud,
            gt_pointcloud,
        ],
        3,
        2,
    )


def load_transpose_intrinsics_if_needed(args, dataset_kind):
    if not args.save_vis or dataset_kind != "transpose":
        return None

    if not os.path.exists(args.intrinsics_path):
        print(f"Error: Intrinsics path '{args.intrinsics_path}' does not exist")
        sys.exit(1)
    if args.pc_knn_k < 1:
        print(f"Error: --pc-knn-k must be greater than 0, got {args.pc_knn_k}")
        sys.exit(1)
    if args.pc_knn_std_ratio < 0:
        print(
            "Error: --pc-knn-std-ratio must be non-negative, "
            f"got {args.pc_knn_std_ratio}"
        )
        sys.exit(1)

    return load_intrinsics(args.intrinsics_path)


def inference(args):
    validate_inputs(args)

    dataset, dataset_kind = load_test_dataset(args.dataset, args.raw_type)
    args.dataset_kind = dataset_kind
    if hasattr(dataset, "depth_scale"):
        args.depth_scale = dataset.depth_scale
    args.min_depth = float(dataset.depth_range[0])
    args.max_depth = float(dataset.depth_range[1])
    dataset = limit_dataset_for_eval(dataset, args.max_samples)
    args.num_samples = len(dataset)
    args.device = DEVICE
    args.resolved_model_module = "promptda.promptda"
    args.resolved_model_class = "PromptDA"
    args.prediction_kind = "metric_depth_meters"
    args.uses_prompt_depth = True

    transpose_intrinsics = load_transpose_intrinsics_if_needed(args, dataset_kind)
    model = load_model(args)

    with open(Path(args.output) / "args.json", "w", encoding="utf-8") as file:
        json.dump(vars(args), file, indent=2)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    dataset_label = dataset_kind.upper()
    for batch_items in tqdm(dataloader, desc=f"PromptDA {dataset_label} inference"):
        if len(batch_items) == 4:
            rgb_paths, raw_depth_paths, gt_depth_paths, sample_names = batch_items
        else:
            rgb_paths, raw_depth_paths, gt_depth_paths = batch_items
            sample_names = None

        for sample_idx, (rgb_path, raw_depth_path, gt_depth_path) in enumerate(zip(
            rgb_paths, raw_depth_paths, gt_depth_paths
        )):
            rgb_path = str(rgb_path)
            raw_depth_path = str(raw_depth_path)
            gt_depth_path = str(gt_depth_path)
            if sample_names is None:
                sample = (rgb_path, raw_depth_path, gt_depth_path)
            else:
                sample = (
                    rgb_path,
                    raw_depth_path,
                    gt_depth_path,
                    sample_names[sample_idx],
                )
            name = sample_name_for_sample(dataset_kind, sample)

            pred, rgb_for_vis, prompt_depth, rgb_original_shape = predict_depth(
                model, rgb_path, raw_depth_path, gt_depth_path, args
            )
            np.save(Path(args.prediction_dir) / f"{name}.npy", pred)

            if args.save_vis and dataset_kind == "transpose":
                gt_depth_for_vis = load_depth_array(
                    gt_depth_path, args.depth_scale, args.max_depth
                )
                scaled_intrinsics = scale_intrinsics(
                    transpose_intrinsics,
                    rgb_original_shape,
                    rgb_for_vis.shape[:2],
                )
                grid_vis = create_transpose_visualization(
                    rgb_for_vis,
                    prompt_depth,
                    pred,
                    gt_depth_for_vis,
                    scaled_intrinsics,
                    args,
                )
                Image.fromarray(grid_vis).save(
                    Path(args.visualization_dir) / f"{name}_grid_vis.jpg"
                )
            elif args.save_vis:
                save_visualization(
                    Path(args.visualization_dir) / f"{name}_promptda_vis.jpg",
                    rgb_for_vis,
                    prompt_depth,
                    pred,
                    args,
                )


if __name__ == "__main__":
    inference(parse_arguments())
