"""Load the vendored Segment Anything model for prompted inference."""

from pathlib import Path
import sys

import torch


def load_sam_predictor(repo, checkpoint, model_type, device):
    source = str(Path(repo) / "segment-anything")
    if source not in sys.path:
        sys.path.insert(0, source)
    from segment_anything import SamPredictor, sam_model_registry

    if checkpoint is None or not Path(checkpoint).is_file():
        raise FileNotFoundError("Set sam.checkpoint to the local SAM weights file.")
    model = sam_model_registry[model_type]()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.to(device).eval().requires_grad_(False)
    return SamPredictor(model)
