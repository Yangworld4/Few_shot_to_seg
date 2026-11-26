"""Integer-grid Euclidean distance transform for Native CRAG evaluation."""

import numpy as np
from numba import njit
from scipy import ndimage

@njit(cache=True)
def _rectangular_edt(mask):
    height, width = mask.shape
    absent = height * height + width * width + 1
    horizontal = np.empty((width, height), np.int32)
    for y in range(height):
        nearest = -width
        for x in range(width):
            if not mask[y, x]:
                nearest = x
            distance = x - nearest
            horizontal[x, y] = distance * distance if nearest >= 0 else absent
        nearest = 2 * width
        for x in range(width - 1, -1, -1):
            if not mask[y, x]:
                nearest = x
            if nearest < width:
                distance = nearest - x
                horizontal[x, y] = min(horizontal[x, y], distance * distance)

    result = np.empty((width, height), dtype=np.float64)
    locations = np.empty(height, dtype=np.int32)
    starts = np.empty(height, dtype=np.int32)
    for x in range(width):
        column = horizontal[x]
        count = 0
        for y in range(height):
            if column[y] == absent:
                continue
            start = 0
            while count:
                previous = locations[count - 1]
                # First integer coordinate where the new parabola is at least as close.
                numerator = column[y] + y * y - column[previous] - previous * previous
                denominator = 2 * (y - previous)
                start = -((-numerator) // denominator)
                if start > starts[count - 1]:
                    break
                count -= 1
            if not count:
                start = 0
            if start < height:
                locations[count] = y
                starts[count] = start
                count += 1
        active = 0
        for y in range(height):
            if active + 1 < count and y == starts[active + 1]:
                active += 1
            dy = y - locations[active]
            result[x, y] = np.sqrt(float(dy * dy + column[locations[active]]))
    return np.ascontiguousarray(result.T)

def exact_edt(mask):
    # Preserve SciPy's empty-feature behavior and avoid integer overflow.
    if (mask.ndim != 2 or mask.dtype != np.bool_ or sum(s * s for s in mask.shape) >= 2**31
            or mask.all()):
        return ndimage.distance_transform_edt(mask)
    return _rectangular_edt(np.ascontiguousarray(mask))
