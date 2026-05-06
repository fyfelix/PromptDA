import argparse
import json
import os
import sys
from datetime import datetime
from os.path import exists, join

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="PromptDA depth evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--encoder",
        type=str,
        choices=["vits", "vitb", "vitl", "vitg"],
        default="vitl",
        help="Model encoder type",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the model checkpoint file",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="HAMMER, ClearPose, or DREDS JSONL path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output_dir",
        help="Directory for evaluation outputs",
    )
    parser.add_argument(
        "--prediction-dir",
        type=str,
        default=None,
        help="Optional directory containing per-sample .npy predictions",
    )
    parser.add_argument(
        "--raw-type",
        type=str,
        required=True,
        choices=["d435", "l515", "tof"],
        help="Raw depth type; ClearPose only supports d435",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=518,
        help="Input size recorded for run metadata",
    )
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=1000.0,
        help="Scale factor for depth values",
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=6.0,
        help="Maximum valid depth value",
    )
    parser.add_argument(
        "--image-min",
        type=float,
        default=0.1,
        help="Minimum visualization depth recorded for run metadata",
    )
    parser.add_argument(
        "--image-max",
        type=float,
        default=5.0,
        help="Maximum visualization depth recorded for run metadata",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Maximum number of dataset samples to evaluate; 0 means all samples",
    )
    return parser.parse_args()


if __name__ == "__main__" and any(arg in ("-h", "--help") for arg in sys.argv[1:]):
    parse_arguments()
    sys.exit(0)


import cv2
import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dataset import (
    limit_dataset_for_eval,
    load_test_dataset,
    sample_name_for_dataset,
)
from utils.metric import (
    abs_relative_difference,
    delta1_acc,
    delta4_acc_105,
    delta5_acc110,
    mae_linear,
    rmse_linear,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_gt_depth(depth_path, depth_scale, max_depth, min_depth):
    depth_gt = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_gt is None:
        raise ValueError(f"Could not load GT depth from {depth_path}")

    depth_gt = np.asarray(depth_gt).astype(np.float32)
    if depth_gt.ndim == 3:
        depth_gt = depth_gt[..., 0]
    depth_gt = depth_gt / depth_scale

    valid_mask = (depth_gt >= min_depth) & (depth_gt <= max_depth)
    depth_gt[~valid_mask] = min_depth
    return depth_gt, valid_mask


def align_prediction_shape(pred, gt_shape, dataset_kind, name):
    if pred.shape == gt_shape:
        return pred

    if dataset_kind != "dreds":
        raise ValueError(
            f"Prediction/GT shape mismatch for {name}: "
            f"dataset_kind={dataset_kind}, pred_shape={pred.shape}, gt_shape={gt_shape}"
        )

    gt_height, gt_width = gt_shape
    return cv2.resize(
        pred.astype(np.float32, copy=False),
        (gt_width, gt_height),
        interpolation=cv2.INTER_NEAREST,
    )


class EvalDataset(Dataset):
    def __init__(self, dataset, output_path, args, depth_scale, align=False):
        self.dataset = dataset
        self.prediction_path = args.prediction_dir or join(output_path, "predictions")
        self.legacy_prediction_path = output_path
        self.args = args
        self.depth_scale = depth_scale
        self.align = align

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        rgb_path, _, gt_depth_path = self.dataset[idx]
        depth_gt, valid_mask = load_gt_depth(
            gt_depth_path,
            self.depth_scale,
            self.args.max_depth,
            self.args.min_depth,
        )

        name = sample_name_for_dataset(self.args.dataset_kind, rgb_path)
        pred_path = join(self.prediction_path, name + ".npy")
        if not exists(pred_path):
            pred_path = join(self.legacy_prediction_path, name + ".npy")
        if not exists(pred_path):
            raise FileNotFoundError(
                f"Prediction for {name} not found in "
                f"{self.prediction_path} or {self.legacy_prediction_path}"
            )

        pred = np.load(pred_path)
        pred = align_prediction_shape(pred, depth_gt.shape, self.args.dataset_kind, name)

        pred_invalid_mask = np.logical_or(np.isnan(pred), np.isinf(pred))
        if pred_invalid_mask.sum() > 0:
            valid_mask = valid_mask & ~pred_invalid_mask

        if self.align:
            depth_gt_reshaped = depth_gt[valid_mask].reshape((-1, 1))
            pred_reshaped = pred[valid_mask].reshape((-1, 1))

            ones = np.ones_like(pred_reshaped)
            matrix = np.concatenate([pred_reshaped, ones], axis=-1)
            scale, shift = np.linalg.lstsq(matrix, depth_gt_reshaped, rcond=None)[0]
            pred_reshaped = scale * pred_reshaped + shift
            pred_reshaped = np.clip(
                pred_reshaped,
                a_min=self.args.min_depth,
                a_max=None,
            )

            return {
                "name": name,
                "pred": pred_reshaped.astype(np.float32),
                "gt": depth_gt_reshaped.astype(np.float32),
                "mask": np.ones_like(pred_reshaped, dtype=bool),
                "is_aligned": True,
            }

        return {
            "name": name,
            "pred": pred.astype(np.float32),
            "gt": depth_gt.astype(np.float32),
            "mask": valid_mask.astype(bool),
            "is_aligned": False,
        }


def main():
    args = parse_arguments()

    if args.max_samples < 0:
        raise ValueError("--max-samples must be >= 0")

    os.makedirs(args.output, exist_ok=True)

    dataset, dataset_kind = load_test_dataset(args.dataset, args.raw_type)
    args.dataset_kind = dataset_kind
    if hasattr(dataset, "depth_scale"):
        args.depth_scale = dataset.depth_scale

    dataset = limit_dataset_for_eval(dataset, args.max_samples)
    args.num_samples = len(dataset)

    depth_scale = args.depth_scale
    args.min_depth = dataset.depth_range[0]
    args.max_depth = dataset.depth_range[1]

    with open(join(args.output, "eval_args.json"), "w", encoding="utf-8") as file:
        json.dump(vars(args), file)

    print(
        "min depth is updated and set to ",
        args.min_depth,
        "and max depth is updated and set to ",
        args.max_depth,
    )
    print(f"evaluation device: {DEVICE}")

    align = False
    eval_dataset = EvalDataset(dataset, args.output, args, depth_scale, align=align)
    batch_size = 1 if align else 32
    num_workers = 0 if align or DEVICE != "cuda" else 8

    loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=DEVICE == "cuda",
    )

    all_metrics = []
    for batch in tqdm(loader):
        names = batch["name"]

        pred_depth_ts = batch["pred"].to(DEVICE)
        gt_depth_ts = batch["gt"].to(DEVICE)
        mask_ts = batch["mask"].to(DEVICE)

        l1 = mae_linear(pred_depth_ts, gt_depth_ts, mask_ts, reduction="none")
        rmse = rmse_linear(pred_depth_ts, gt_depth_ts, mask_ts, reduction="none")
        abs_rel = abs_relative_difference(
            pred_depth_ts,
            gt_depth_ts,
            mask_ts,
            reduction="none",
        )
        d4 = delta4_acc_105(pred_depth_ts, gt_depth_ts, mask_ts, reduction="none")
        d5 = delta5_acc110(pred_depth_ts, gt_depth_ts, mask_ts, reduction="none")
        d1 = delta1_acc(pred_depth_ts, gt_depth_ts, mask_ts, reduction="none")

        batch_len = len(names)
        l1_cpu = l1.detach().cpu().numpy()
        rmse_cpu = rmse.detach().cpu().numpy()
        abs_rel_cpu = abs_rel.detach().cpu().numpy()
        d4_cpu = d4.detach().cpu().numpy()
        d5_cpu = d5.detach().cpu().numpy()
        d1_cpu = d1.detach().cpu().numpy()

        for i in range(batch_len):
            all_metrics.append(
                {
                    "name": names[i],
                    "L1": l1_cpu[i],
                    "rmse_linear": rmse_cpu[i],
                    "abs_relative_difference": abs_rel_cpu[i],
                    "delta4_acc_105": d4_cpu[i],
                    "delta5_acc110": d5_cpu[i],
                    "delta1_acc": d1_cpu[i],
                }
            )

    all_metrics = pd.DataFrame(all_metrics)
    all_metrics_mean = all_metrics.mean(numeric_only=True).to_frame().T

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    all_metrics.to_csv(
        join(args.output, f"all_metrics_{current_time}_{align}.csv"),
        index=False,
    )
    all_metrics_mean.to_json(
        join(args.output, f"mean_metrics_{current_time}_{align}.json"),
        orient="records",
        lines=True,
        force_ascii=False,
    )
    logger.info(f"save dir: {args.output}")


if __name__ == "__main__":
    main()
