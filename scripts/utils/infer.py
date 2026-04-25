import matplotlib
import numpy as np

from scripts.utils.test_datasets import load_images


def colorize_depth_maps(depth_map, min_depth, max_depth, cmap="Spectral", valid_mask=None):
    assert len(depth_map.shape) >= 2, "Invalid dimension"

    depth = depth_map.copy().squeeze()
    if depth.ndim < 3:
        depth = depth[np.newaxis, :, :]

    cm_func = matplotlib.colormaps[cmap]
    depth = ((depth - min_depth) / (max_depth - min_depth)).clip(0, 1)
    img_colored_np = cm_func(depth, bytes=False)[:, :, :, 0:3]
    img_colored_np = np.rollaxis(img_colored_np, 3, 1)

    if valid_mask is not None:
        valid_mask = valid_mask.squeeze()
        if valid_mask.ndim < 3:
            valid_mask = valid_mask[np.newaxis, np.newaxis, :, :]
        else:
            valid_mask = valid_mask[:, np.newaxis, :, :]
        valid_mask = np.repeat(valid_mask, 3, axis=1)
        img_colored_np[~valid_mask] = 0

    return img_colored_np
