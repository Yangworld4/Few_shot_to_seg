import csv
from pathlib import Path
import numpy as np
from PIL import Image


def read_manifest(path, root=None, require_mask=False):
    if path is None:
        raise ValueError("A manifest path is required for this action")
    path = Path(path).expanduser().resolve()
    base = Path(root).expanduser().resolve() if root else path.parent
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    entries = []
    ids = set()
    for row in rows:
        image = row.get("image") or row.get("image_path")
        mask = row.get("mask") or row.get("mask_path")
        if not image or require_mask and not mask:
            raise ValueError("Manifest requires image and, for support/evaluation, mask columns")
        image_path = (base / image).resolve()
        mask_path = (base / mask).resolve() if mask else None
        image_id = row.get("image_id") or image_path.stem
        if image_id in ids:
            raise ValueError(f"Duplicate image ID: {image_id}")
        ids.add(image_id)
        if not image_path.is_file() or mask_path and not mask_path.is_file():
            raise FileNotFoundError(f"Missing image or mask for {image_id}")
        entries.append({"id": image_id, "image": image_path, "mask": mask_path,
                        "instance_mask": (base / row["instance_mask"]).resolve() if row.get("instance_mask") else None})
    return entries


def load_image(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def load_mask(path, config):
    with Image.open(path) as im:
        if config["mask_encoding"] == "rgb":
            raw = np.asarray(im.convert("RGB"))
        else:
            raw = np.asarray(im)
            if raw.ndim != 2:
                raise ValueError("Indexed mask must be single channel; configure RGB mapping explicitly")
    mapping = config["class_mapping"]
    if not mapping:
        raise ValueError("dataset.class_mapping is required")
    output = np.full(raw.shape[:2], -1, np.int32)
    for value, cls in mapping.items():
        if config["mask_encoding"] == "rgb":
            color = tuple(int(v) for v in str(value).split(","))
            if len(color) != 3:
                raise ValueError("RGB mapping keys must be 'R,G,B'")
            match = np.all(raw == color, axis=-1)
        else:
            match = raw == int(value)
        output[match] = int(cls)
    if np.any(output < 0):
        raise ValueError(f"Mask contains unmapped labels at {path}")
    return output


def resize_mask(mask, shape):
    if mask.shape == tuple(shape):
        return mask
    import cv2
    # Match the validated Direct evaluation's nearest-neighbor GT resize.
    return cv2.resize(mask.astype(np.int32), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
