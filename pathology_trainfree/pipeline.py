from pathlib import Path
import hashlib
import numpy as np
import torch
from PIL import Image
from .geometry import windows, reconstruct, resize_map
from .retrieval import SimilarityRetriever


class SegmentationPipeline:
    def __init__(self, encoder, memory, cfg, feature_store=None):
        self.feature_store = feature_store
        self.encoder = encoder
        self.memory = memory
        self.cfg = cfg
        memory.validate_for(cfg, encoder)
        self.retriever = SimilarityRetriever(memory, cfg, encoder.device)

    @torch.inference_mode()
    def predict(self, image, image_id=None):
        w = self.cfg["window"]
        s = self.cfg["segmentation"]
        ids = self.retriever.ids
        h, width = image.shape[:2]
        grids = w["size"] // self.encoder.patch_size
        offset_views = []
        for offset in w["offsets"]:
            views = list(windows(image, w["size"], w["stride"], offset, w["padding_mode"]))
            positions = [(y, x) for y, x, _ in views]
            cache_dir = self.cfg["output"].get("directory")
            cache_file = None
            cache_key = None
            if self.cfg["output"].get("cache_features") and image_id and cache_dir:
                cache_file = Path(cache_dir) / "feature_cache" / f"{image_id}_offset{offset}.npz"
                cache_key = hashlib.sha256(image.tobytes() + str((self.encoder.weights_sha256, w["size"],
                                        w["stride"], offset, self.encoder.precision,
                                        self.encoder.token_precision)).encode()).hexdigest()
            if self.feature_store is not None:
                tokens, _ = self.feature_store.get(image, image_id, offset)
            elif cache_file and cache_file.exists():
                with np.load(cache_file) as z:
                    tokens = z["tokens"]
                    cached_positions = z["positions"]
                    stored_key = str(z["cache_key"])
                if stored_key != cache_key:
                    raise ValueError("Cached query features differ from current image or encoder configuration")
                if not np.array_equal(cached_positions, positions):
                    raise ValueError("Cached feature positions differ from current window settings")
            else:
                tokens = self.encoder.encode([crop for _, _, crop in views])
                if cache_file:
                    cache_file.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(cache_file, tokens=tokens, positions=np.asarray(positions), cache_key=np.array(cache_key))
            scores = self.retriever.scores(tokens).numpy().reshape(len(views), grids, grids, len(ids))
            if ids == [0, 1]:
                # Match Direct: score difference is rounded through fp16 before interpolation.
                maps = scores[..., 1] - scores[..., 0]
                offset_views.append(reconstruct(maps, positions, (h, width), w["size"], s["margin_precision"]))
            else:
                channels = [reconstruct(scores[..., j], positions, (h, width), w["size"], s["margin_precision"]) for j in range(len(ids))]
                offset_views.append(np.stack(channels, axis=0))
        native = np.mean(np.stack(offset_views), axis=0, dtype=np.float32)
        output_size = "native" if s["native_resolution"] else s["output_size"]
        if output_size == "native":
            output = native
        elif ids == [0, 1]:
            output = resize_map(native, output_size)
        else:
            output = np.stack([resize_map(channel, output_size) for channel in native])
        if ids == [0, 1]:
            mask = (output > float(s["threshold"])).astype(np.uint8)
            return {"mask": mask, "native_scores": native, "scores": output}
        mask = np.asarray(ids, dtype=np.int32)[output.argmax(axis=0)]
        return {"mask": mask, "native_scores": native, "scores": output}


def save_prediction(result, image, image_id, cfg):
    out = Path(cfg["output"]["directory"])
    out.mkdir(parents=True, exist_ok=True)
    if cfg["output"]["save_mask"]:
        mask = result["mask"]
        if mask.max() > 255 or mask.min() < 0:
            np.save(out / f"{image_id}_mask.npy", mask)
        else:
            Image.fromarray(mask.astype(np.uint8)).save(out / f"{image_id}_mask.png")
    if cfg["output"]["save_margin"]:
        np.save(out / f"{image_id}_scores.npy", result["scores"].astype(np.float32))
        np.save(out / f"{image_id}_native_scores.npy", result["native_scores"].astype(np.float32))
    if cfg["output"]["save_visualization"]:
        rgb = Image.fromarray(image).resize((result["mask"].shape[1], result["mask"].shape[0]))
        overlay = np.asarray(rgb).copy()
        fg = result["mask"] != 0
        overlay[fg] = (overlay[fg].astype(np.float32) * 0.6 + np.array([255, 0, 0]) * 0.4).astype(np.uint8)
        Image.fromarray(overlay).save(out / f"{image_id}_overlay.png")
