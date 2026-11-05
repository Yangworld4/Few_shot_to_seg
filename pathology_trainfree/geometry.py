import math
import numpy as np
import torch
import torch.nn.functional as F


def starts(length, size, stride, offset):
    n = max(1, math.ceil((length + offset - size) / stride) + 1)
    return [-offset + stride * i for i in range(n)]


def windows(image, size, stride, offset, mode="reflect"):
    h, w = image.shape[:2]
    for y in starts(h, size, stride, offset):
        for x in starts(w, size, stride, offset):
            top, left = max(0, -y), max(0, -x)
            bottom, right = max(0, y + size - h), max(0, x + size - w)
            crop = image[max(0, y):min(h, y + size), max(0, x):min(w, x + size)]
            # np.pad reflect requires a source extent >1. Reject impossible images explicitly.
            if (top or bottom) and crop.shape[0] < 2 or (left or right) and crop.shape[1] < 2:
                raise ValueError("Reflect padding needs at least two source pixels per padded axis")
            pad = ((top, bottom), (left, right)) + (((0, 0),) if image.ndim == 3 else ())
            crop = np.pad(crop, pad, mode=mode)
            if crop.shape[:2] != (size, size):
                raise RuntimeError("Window geometry produced an incorrect crop")
            yield y, x, crop


def resize_map(values, shape):
    t = torch.as_tensor(np.asarray(values, dtype=np.float32))[None, None]
    return F.interpolate(t, size=tuple(shape), mode="bilinear", align_corners=False)[0, 0].numpy()


def reconstruct(grids, positions, image_shape, size, value_precision="float16"):
    h, w = image_shape
    total = np.zeros((h, w), np.float32)
    count = np.zeros((h, w), np.float32)
    for grid, (y, x) in zip(grids, positions):
        dtype = np.float16 if value_precision == "float16" else np.float32
        expanded = resize_map(np.asarray(grid, dtype=dtype).astype(np.float32), (size, size))
        y0, x0, y1, x1 = max(0, y), max(0, x), min(h, y + size), min(w, x + size)
        total[y0:y1, x0:x1] += expanded[y0-y:y1-y, x0-x:x1-x]
        count[y0:y1, x0:x1] += 1
    if not np.all(count > 0):
        raise RuntimeError("Uncovered pixels in window reconstruction")
    return total / count
