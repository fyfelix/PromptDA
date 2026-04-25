from PIL import Image
import numpy as np
import torch


def concat_images(images, output_path=None, direction="horizontal"):
    if not images or len(images) < 2:
        raise ValueError("At least two images are required for concatenation.")

    if not isinstance(images[0], Image.Image):
        images = [Image.open(img) for img in images]

    if direction == "horizontal":
        total_width = sum(img.width for img in images)
        max_height = max(img.height for img in images)
        output = Image.new("RGB", (total_width, max_height))
        x_offset = 0
        for img in images:
            output.paste(img, (x_offset, 0))
            x_offset += img.width
    elif direction == "vertical":
        max_width = max(img.width for img in images)
        total_height = sum(img.height for img in images)
        output = Image.new("RGB", (max_width, total_height))
        y_offset = 0
        for img in images:
            output.paste(img, (0, y_offset))
            y_offset += img.height
    else:
        raise ValueError("Direction must be 'horizontal' or 'vertical'.")

    if output_path is not None:
        output.save(output_path)

    return output


def depth2disparity(depth, return_mask=False):
    if isinstance(depth, torch.Tensor):
        disparity = torch.zeros_like(depth)
    elif isinstance(depth, np.ndarray):
        disparity = np.zeros_like(depth)
    else:
        raise TypeError(f"Unsupported depth type: {type(depth)}")

    valid_mask = depth > 0
    disparity[valid_mask] = 1.0 / depth[valid_mask]
    if return_mask:
        return disparity, valid_mask
    return disparity


def disparity2depth(disparity, return_mask=False):
    return depth2disparity(disparity, return_mask=return_mask)
