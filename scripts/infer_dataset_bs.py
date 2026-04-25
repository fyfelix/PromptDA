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
from torch.utils.data import DataLoader
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from promptda.promptda import PromptDA
from scripts.utils.test_datasets import (
    ClearPoseDataset,
    HAMMERDataset,
    load_images,
    prediction_name,
)


DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="PromptDA batch inference for the HAMMER/ClearPose benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-path", type=str, required=True, help="Path to model.ckpt")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the JSONL dataset")
    parser.add_argument("--output", type=str, default="output_dir", help="Output directory")
    parser.add_argument(
        "--raw-type",
        type=str,
        required=True,
        choices=["d435", "l515", "tof"],
        help="Camera raw depth type",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="vitl",
        choices=["vits", "vitb", "vitl", "vitg"],
        help="PromptDA encoder architecture",
    )
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--batch-size", "-b", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-depth", type=float, default=6.0)
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument(
        "--input-size",
        dest="input_size",
        type=int,
        default=1008,
        help="RGB resize bound before PromptDA inference; set <=0 to only enforce /14 size",
    )
    parser.add_argument(
        "--max-size",
        dest="input_size",
        type=int,
        default=argparse.SUPPRESS,
        help="Deprecated alias for --input-size",
    )
    parser.add_argument(
        "--prediction-resize-mode",
        type=str,
        default="bilinear",
        choices=["bilinear", "nearest"],
        help="Interpolation mode used to resize predictions back to GT resolution",
    )
    parser.add_argument(
        "--resize-method",
        type=str,
        default="upper_bound",
        choices=["upper_bound", "lower_bound", "minimal"],
        help="RGB resize rule, matching the benchmark Resize transform naming",
    )
    return parser.parse_args()


def validate_inputs(args):
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model path does not exist: {args.model_path}")
    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"Dataset file does not exist: {args.dataset}")
    os.makedirs(args.output, exist_ok=True)


def load_dataset(args):
    dataset_name = args.dataset.lower()
    if "clearpose" in dataset_name:
        if args.raw_type != "d435":
            raise ValueError("ClearPose dataset only supports d435 raw type")
        return ClearPoseDataset(args.dataset)
    if "hammer" in dataset_name:
        return HAMMERDataset(args.dataset, args.raw_type)
    raise ValueError(f"Invalid dataset: {args.dataset}")


def batch_collate(batch):
    rgb_paths = [item[0] for item in batch]
    raw_depth_paths = [item[1] for item in batch]
    gt_depth_paths = [item[2] for item in batch]
    return rgb_paths, raw_depth_paths, gt_depth_paths


def ensure_multiple_of(value, multiple_of=14):
    return max(multiple_of, int(value // multiple_of * multiple_of))


def constrain_to_multiple_of(value, multiple_of=14, min_val=0, max_val=None):
    target = int(np.floor(value / multiple_of) * multiple_of)
    if max_val is not None and target > max_val:
        target = int(np.floor(max_val / multiple_of) * multiple_of)
    if target < min_val:
        target = int(np.ceil(value / multiple_of) * multiple_of)
    return max(multiple_of, target)


def get_resized_size(width, height, input_size, resize_method, multiple_of=14):
    if input_size is None or input_size <= 0:
        return (
            ensure_multiple_of(width, multiple_of),
            ensure_multiple_of(height, multiple_of),
        )

    scale_height = input_size / height
    scale_width = input_size / width

    if resize_method == "lower_bound":
        scale = max(scale_width, scale_height)
        max_val = None
        min_val = input_size
    elif resize_method == "upper_bound":
        scale = min(scale_width, scale_height)
        max_val = input_size
        min_val = 0
    elif resize_method == "minimal":
        scale = (
            scale_width
            if abs(1.0 - scale_width) < abs(1.0 - scale_height)
            else scale_height
        )
        max_val = None
        min_val = 0
    else:
        raise ValueError(f"Invalid resize_method: {resize_method}")

    target_w = constrain_to_multiple_of(
        scale * width,
        multiple_of=multiple_of,
        min_val=min_val,
        max_val=max_val,
    )
    target_h = constrain_to_multiple_of(
        scale * height,
        multiple_of=multiple_of,
        min_val=min_val,
        max_val=max_val,
    )
    return target_w, target_h


def resize_rgb_for_promptda(rgb, input_size, resize_method="upper_bound", multiple_of=14):
    height, width = rgb.shape[:2]

    if (resize_method == "upper_bound"
            and input_size is not None and input_size > 0
            and max(height, width) <= input_size):
        target_h = ensure_multiple_of(height, multiple_of)
        target_w = ensure_multiple_of(width, multiple_of)
    else:
        target_w, target_h = get_resized_size(width, height, input_size, resize_method, multiple_of)

    if target_h == height and target_w == width:
        return rgb

    scale = max(target_h / height, target_w / width)
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(rgb, (target_w, target_h), interpolation=interpolation)


def load_gt_shape(gt_path):
    gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
    if gt is None:
        raise ValueError(f"Could not load GT depth from {gt_path}")
    return gt.shape[:2]


def to_image_tensor(rgb):
    image = rgb.astype(np.float32) / 255.0
    return torch.from_numpy(image).permute(2, 0, 1).float()


def to_depth_tensor(depth):
    return torch.from_numpy(depth.astype(np.float32)).unsqueeze(0).float()


def resize_prediction(pred_depth, target_shape, mode="bilinear"):
    if pred_depth.ndim == 3:
        pred_depth = pred_depth.squeeze(0)

    resize_kwargs = {
        "size": target_shape,
        "mode": mode,
    }
    if mode == "bilinear":
        resize_kwargs["align_corners"] = False

    pred_depth = F.interpolate(pred_depth[None, None], **resize_kwargs)[0, 0]
    return pred_depth.detach().cpu().numpy().astype(np.float32)


@torch.no_grad()
def predict_batch(model, images, prompt_depths):
    image_batch = torch.stack(images).to(DEVICE)
    prompt_batch = torch.stack(prompt_depths).to(DEVICE)
    pred_depths = model.predict(image_batch, prompt_batch)

    if pred_depths.ndim == 4:
        pred_depths = pred_depths[:, 0]
    elif pred_depths.ndim != 3:
        raise ValueError(f"Unexpected prediction shape: {tuple(pred_depths.shape)}")

    return pred_depths


def run_single(model, image, prompt_depth):
    return predict_batch(model, [image], [prompt_depth])[0]


def load_model(args):
    print(f"Loading PromptDA {args.encoder} from {args.model_path}")
    model = PromptDA.from_pretrained(
        args.model_path,
        model_kwargs={"encoder": args.encoder},
    )
    model = model.to(DEVICE).eval()
    print(f"Model loaded on {DEVICE}")
    return model


@torch.no_grad()
def inference(args):
    validate_inputs(args)
    with open(os.path.join(args.output, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    model = load_model(args)
    dataset = load_dataset(args)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=batch_collate,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    for rgb_paths, raw_depth_paths, gt_depth_paths in tqdm(
        dataloader, desc="Processing batches"
    ):
        records = []
        for rgb_path, raw_depth_path, gt_depth_path in zip(
            rgb_paths, raw_depth_paths, gt_depth_paths
        ):
            name = prediction_name(rgb_path, args.dataset)
            try:
                rgb_src, prompt_depth = load_images(
                    rgb_path, raw_depth_path, args.depth_scale, args.max_depth
                )
                rgb_input = resize_rgb_for_promptda(
                    rgb_src,
                    args.input_size,
                    resize_method=args.resize_method,
                )
                target_shape = load_gt_shape(gt_depth_path)

                h_rgb, w_rgb = rgb_input.shape[:2]
                if prompt_depth.shape[:2] != (h_rgb, w_rgb):
                    prompt_depth = cv2.resize(
                        prompt_depth, (w_rgb, h_rgb),
                        interpolation=cv2.INTER_NEAREST,
                    )

                records.append(
                    {
                        "name": name,
                        "image": to_image_tensor(rgb_input),
                        "prompt_depth": to_depth_tensor(prompt_depth),
                        "target_shape": target_shape,
                    }
                )
            except Exception as exc:
                print(f"Failed to load sample {rgb_path}: {exc}")

        if not records:
            continue

        try:
            pred_depths = predict_batch(
                model,
                [record["image"] for record in records],
                [record["prompt_depth"] for record in records],
            )
            for pred_depth, record in zip(pred_depths, records):
                pred = resize_prediction(
                    pred_depth,
                    record["target_shape"],
                    mode=args.prediction_resize_mode,
                )
                np.save(os.path.join(args.output, f"{record['name']}.npy"), pred)
        except RuntimeError as exc:
            print(f"Batch shape mismatch or OOM, falling back to single-sample mode: {exc}")
            for record in records:
                pred_depth = run_single(model, record["image"], record["prompt_depth"])
                pred = resize_prediction(
                    pred_depth,
                    record["target_shape"],
                    mode=args.prediction_resize_mode,
                )
                np.save(os.path.join(args.output, f"{record['name']}.npy"), pred)


if __name__ == "__main__":
    inference(parse_arguments())
