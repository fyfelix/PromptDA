#!/usr/bin/env python3

import argparse
import json
import os
import sys
from pathlib import Path

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

from dataset import HAMMERDataset
from promptda.promptda import PromptDA


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


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
        description="PromptDA inference for HAMMER evaluation",
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
        "--dataset", type=str, required=True, help="HAMMER JSONL path"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output_dir",
        help="Directory for run metadata; also used for predictions/visualizations when dedicated dirs are not set",
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
        help="HAMMER raw depth field used as PromptDA prompt depth",
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
        "--clamp-prediction",
        type=str2bool,
        default=False,
        help="Clamp saved predictions to the HAMMER depth range",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Maximum number of dataset samples to run; 0 means all samples",
    )
    return parser.parse_args()


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
        args.prediction_dir = args.output
    if args.visualization_dir is None:
        args.visualization_dir = args.output

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.prediction_dir, exist_ok=True)
    if args.save_vis:
        os.makedirs(args.visualization_dir, exist_ok=True)


def limit_dataset(dataset, max_samples):
    if max_samples > 0:
        dataset.data = dataset.data[:max_samples]
    return dataset


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
        raise ValueError(f"Could not load GT depth image: {gt_depth_path}")
    return gt_depth.shape[:2]


def sample_id_from_rgb_path(rgb_path):
    parts = rgb_path.split("/")
    scene_name = parts[-4]
    frame_name = Path(parts[-1]).stem
    return f"{scene_name}#{frame_name}"


@torch.no_grad()
def predict_depth(model, rgb_path, raw_depth_path, gt_depth_path, args):
    rgb_tensor, _, rgb_for_vis = load_rgb_tensor(rgb_path, args.input_size)
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

    return pred, rgb_for_vis, prompt_depth


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


def inference(args):
    validate_inputs(args)

    dataset = HAMMERDataset(args.dataset, args.raw_type)
    args.min_depth = float(dataset.depth_range[0])
    args.max_depth = float(dataset.depth_range[1])
    dataset = limit_dataset(dataset, args.max_samples)
    args.num_samples = len(dataset)
    args.device = DEVICE
    args.resolved_model_module = "promptda.promptda"
    args.resolved_model_class = "PromptDA"
    args.prediction_kind = "metric_depth_meters"
    args.uses_prompt_depth = True

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

    for batch_items in tqdm(dataloader, desc="PromptDA HAMMER inference"):
        rgb_paths, raw_depth_paths, gt_depth_paths = batch_items
        for rgb_path, raw_depth_path, gt_depth_path in zip(
            rgb_paths, raw_depth_paths, gt_depth_paths
        ):
            rgb_path = str(rgb_path)
            raw_depth_path = str(raw_depth_path)
            gt_depth_path = str(gt_depth_path)
            name = sample_id_from_rgb_path(rgb_path)

            pred, rgb_for_vis, prompt_depth = predict_depth(
                model, rgb_path, raw_depth_path, gt_depth_path, args
            )
            np.save(Path(args.prediction_dir) / f"{name}.npy", pred)

            if args.save_vis:
                save_visualization(
                    Path(args.visualization_dir) / f"{name}_promptda_vis.jpg",
                    rgb_for_vis,
                    prompt_depth,
                    pred,
                    args,
                )


if __name__ == "__main__":
    inference(parse_arguments())
