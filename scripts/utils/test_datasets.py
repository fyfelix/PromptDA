import json
import os
from glob import glob
from os.path import dirname, join

import cv2
import numpy as np
from torch.utils.data import Dataset


MAX_RETRIES = 1000


def load_images(rgb_path, depth_path, depth_scale, max_depth):
    """Load benchmark RGB and raw depth for PromptDA inference.

    The raw depth input is kept as metric depth in meters. Invalid values are
    encoded as 0.0, matching the HAMMER/ClearPose benchmark loaders.
    """
    rgb_src = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if rgb_src is None:
        raise ValueError(f"Could not load RGB image from {rgb_path}")
    rgb_src = np.asarray(rgb_src[:, :, ::-1])

    depth_low_res = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_low_res is None:
        raise ValueError(f"Could not load depth image from {depth_path}")

    if depth_low_res.ndim == 3:
        depth_low_res = depth_low_res[:, :, 0]

    depth_low_res = np.asarray(depth_low_res).astype(np.float32) / depth_scale
    invalid_mask = (
        ~np.isfinite(depth_low_res)
        | (depth_low_res <= 0.0)
        | (depth_low_res > max_depth)
    )
    depth_low_res[invalid_mask] = 0.0

    return rgb_src, depth_low_res


def prediction_name(rgb_path, dataset_path):
    parts = rgb_path.split("/")
    dataset_name = dataset_path.lower()

    if "hammer" in dataset_name:
        scene_name = parts[-4]
        return scene_name + "#" + os.path.splitext(parts[-1])[0]
    if "clearpose" in dataset_name:
        return "#".join(parts[-3:-1]) + "#" + os.path.splitext(parts[-1])[0]

    raise ValueError(f"Invalid dataset: {dataset_path}")


class HAMMERDataset(Dataset):
    def __init__(self, jsonl_path, raw_type="d435"):
        self.jsonl_path = jsonl_path
        self.root = dirname(jsonl_path)
        self.data = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))

        self.raw_type = raw_type
        self.depth_range = self.data[0].get("depth-range", [0.1, 6.0])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        for attempt in range(MAX_RETRIES + 1):
            try:
                item = self.data[idx]
                rgb = join(self.root, item["rgb"])

                raw_type = self.raw_type.lower()
                if raw_type == "d435":
                    raw_depth = join(self.root, item["d435_depth"])
                elif raw_type == "l515":
                    raw_depth = join(self.root, item["l515_depth"])
                elif raw_type == "tof":
                    raw_depth = join(self.root, item["tof_depth"])
                else:
                    raise ValueError(f"Invalid raw type: {self.raw_type}")

                gt_depth = join(self.root, item["depth"])

                if not (
                    os.path.exists(rgb)
                    and os.path.exists(raw_depth)
                    and os.path.exists(gt_depth)
                ):
                    raise FileNotFoundError(f"Missing file(s) for sample {idx}")

                return rgb, raw_depth, gt_depth
            except Exception as exc:
                if attempt < MAX_RETRIES:
                    print(
                        f"Error loading sample {idx} in HAMMERDataset: {exc}. "
                        f"Retrying {attempt + 1}/{MAX_RETRIES}..."
                    )
                    idx = np.random.randint(0, len(self.data))
                else:
                    print("Failed to load sample after retries.")
                    raise


class ClearPoseDataset(Dataset):
    def __init__(self, jsonl_path, max_length_each_sequence=300):
        self.jsonl_path = jsonl_path
        self.root = dirname(jsonl_path)
        self.data = []
        self.rgbs = []
        self.raw_depths = []
        self.gt_depths = []

        depth_range = None
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                if depth_range is None:
                    depth_range = item["depth-range"]

                rgb = sorted(
                    glob(join(self.root, item["rgb"], "*" + item["rgb-suffix"]))
                )[:max_length_each_sequence]
                raw_depth = sorted(
                    glob(join(self.root, item["rgb"], "*" + item["raw_depth-suffix"]))
                )[:max_length_each_sequence]
                gt_depth = sorted(
                    glob(join(self.root, item["rgb"], "*" + item["depth-suffix"]))
                )[:max_length_each_sequence]

                self.rgbs.extend(rgb)
                self.raw_depths.extend(raw_depth)
                self.gt_depths.extend(gt_depth)
                self.data.append(item)

        self.depth_range = depth_range

    def __len__(self):
        return len(self.rgbs)

    def __getitem__(self, idx):
        for attempt in range(MAX_RETRIES + 1):
            try:
                rgb = self.rgbs[idx]
                raw_depth = self.raw_depths[idx]
                gt_depth = self.gt_depths[idx]

                if not (
                    os.path.exists(rgb)
                    and os.path.exists(raw_depth)
                    and os.path.exists(gt_depth)
                ):
                    raise FileNotFoundError(f"Missing file(s) for sample {idx}")

                return rgb, raw_depth, gt_depth
            except Exception as exc:
                if attempt < MAX_RETRIES:
                    print(
                        f"Error loading sample {idx} in ClearPoseDataset: {exc}. "
                        f"Retrying {attempt + 1}/{MAX_RETRIES}..."
                    )
                    idx = np.random.randint(0, len(self.rgbs))
                else:
                    print("Failed to load sample after retries.")
                    raise
