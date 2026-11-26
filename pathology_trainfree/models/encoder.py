"""Extract raw patch tokens from a frozen UNI transformer block."""

from pathlib import Path

import numpy as np
import timm
import torch

from ..utils import sha256


class UNIEncoder:
    def __init__(self, weights, layer, device):
        if weights is None or not Path(weights).is_file():
            raise FileNotFoundError("Set encoder.weights to an existing UNI checkpoint.")
        self.device = torch.device(device)
        self.layer = layer
        self.weights_sha256 = sha256(weights)
        self.model = timm.create_model("vit_large_patch16_224", pretrained=False, dynamic_img_size=True,
                                       num_classes=0, init_values=1e-5)
        self.model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
        self.model.to(self.device).eval().requires_grad_(False)
        self.mean = torch.tensor([.485, .456, .406], device=device)[None, :, None, None]
        self.std = torch.tensor([.229, .224, .225], device=device)[None, :, None, None]

    @torch.inference_mode()
    def encode(self, crops):
        features = []

        def capture(_layer, _args, output):
            patches = output[:, self.model.num_prefix_tokens:]
            features.append(patches.to(device="cpu", dtype=torch.float16).numpy())

        hook = self.model.blocks[self.layer - 1].register_forward_hook(capture)
        try:
            for crop in crops:
                image = torch.from_numpy(crop.copy()).permute(2, 0, 1)[None].to(self.device).float() / 255
                self.model.forward_features((image - self.mean) / self.std)
        finally:
            hook.remove()
        return np.concatenate(features)
