"""Generate gland prompts from a margin map and gate refinements by mask IoU."""

import numpy as np
from scipy import ndimage as ndi


CONNECTIVITY = np.ones((3, 3), dtype=bool)


def split_instances(mask, radius):
    """Split thin bridges by opening, assigning eroded pixels back to the nearest core."""
    labels, count = ndi.label(mask, structure=CONNECTIVITY)
    if not radius:
        return labels, count
    axis = np.arange(-radius, radius + 1)
    disk = axis[:, None] ** 2 + axis[None, :] ** 2 <= radius ** 2
    opened = ndi.binary_opening(mask, structure=disk)
    for label, region in enumerate(ndi.find_objects(labels), 1):
        part = labels[region]
        original = part == label
        cores, ncores = ndi.label(opened[region] & original, structure=CONNECTIVITY)
        if ncores > 1:
            nearest = ndi.distance_transform_edt(cores == 0, return_distances=False, return_indices=True)
            assigned = cores[tuple(nearest)][original]
            part[original] = np.where(assigned == 1, label, assigned + count - 1)
            count += ncores - 1
    return labels, count


def cleanup_mask(mask, max_hole_area, min_area):
    if max_hole_area:
        holes, _ = ndi.label(ndi.binary_fill_holes(mask) & ~mask)
        area = np.bincount(holes.ravel())
        mask = mask | ((holes != 0) & (area[holes] <= max_hole_area))
    if min_area:
        labels, _ = ndi.label(mask, structure=CONNECTIVITY)
        area = np.bincount(labels.ravel())
        mask = (labels != 0) & (area[labels] >= min_area)
    return mask


def _prompt(instance, region, margin, base, threshold, cfg):
    sy, sx = region
    distance = ndi.distance_transform_edt(np.pad(instance, 1))[1:-1, 1:-1]
    py, px = np.unravel_index(distance.argmax(), distance.shape)
    points = [(int(px) + sx.start, int(py) + sy.start)]
    radius = cfg["negative_radius"]
    y0, y1 = max(0, sy.start - radius), min(base.shape[0], sy.stop + radius)
    x0, x1 = max(0, sx.start - radius), min(base.shape[1], sx.stop + radius)
    vicinity = np.s_[y0:y1, x0:x1]
    candidate = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    candidate[sy.start-y0:sy.stop-y0, sx.start-x0:sx.stop-x0] = instance
    near = ndi.distance_transform_edt(~candidate) <= radius
    scores = margin[vicinity]
    ys, xs = np.nonzero(near & ~base[vicinity] & (scores < threshold))
    spacing2 = cfg["negative_spacing"] ** 2
    for _ in range(cfg["negative_points"]):
        if not len(ys):
            break
        index = scores[ys, xs].argmin()
        y, x = ys[index], xs[index]
        points.append((int(x) + x0, int(y) + y0))
        keep = ((ys - y) ** 2 + (xs - x) ** 2 >= spacing2)
        keep[index] = False
        ys, xs = ys[keep], xs[keep]
    box = np.array([sx.start, sy.start, sx.stop - 1, sy.stop - 1], dtype=np.float32)
    return box, np.asarray(points, dtype=np.float32), np.asarray([1] + [0] * (len(points) - 1))


def refine_instances(predictor, margin, base, threshold, cfg):
    labels, _ = split_instances(base, cfg["opening_radius"])
    refined = base.copy()
    attempted = accepted = 0
    for label, region in enumerate(ndi.find_objects(labels), 1):
        instance = labels[region] == label
        area = np.count_nonzero(instance)
        if area < max(1, cfg["min_area"]):
            continue
        box, points, flags = _prompt(instance, region, margin, base, threshold, cfg)
        masks, scores, _ = predictor.predict(point_coords=points, point_labels=flags, box=box,
                                             multimask_output=True)
        attempted += 1
        intersections = np.count_nonzero(masks[:, region[0], region[1]] & instance, axis=(1, 2))
        unions = np.count_nonzero(masks, axis=(1, 2)) + area - intersections
        px, py = points[0].astype(int)
        ious = np.where(masks[:, py, px], intersections / unions, -1)
        best = np.lexsort((scores, ious))[-1]
        if ious[best] >= cfg["iou_threshold"]:
            refined_region = refined[region]
            refined_region[instance] = masks[best][region][instance]
            refined |= masks[best] & ~base
            accepted += 1
    return refined, {"sam_attempted": attempted, "sam_accepted": accepted}
