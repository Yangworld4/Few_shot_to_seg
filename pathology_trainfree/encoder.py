import hashlib
import json
from contextlib import nullcontext
from pathlib import Path
import torch
import torch.nn.functional as F
import timm


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for part in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


class FeatureEncoder:
    """Frozen pretrained encoder with one dense spatial-token interface."""

    patch_size = 16
    feature_dim = 1024

    def __init__(self, config):
        self.name = config["name"]
        specs = {"uni": (16, 1024), "uni2_h": (14, 1536), "kaiko_vitb8": (8, 768)}
        if self.name not in specs:
            raise ValueError(f"Unknown encoder: {self.name}")
        self.patch_size, self.feature_dim = specs[self.name]
        weights = Path(config["weights"]).expanduser().resolve() if config["weights"] else None
        if weights is None or not weights.is_file():
            raise FileNotFoundError("encoder.weights must point to real local pretrained weights")
        config_path = Path(config["config"]).expanduser().resolve() if config.get("config") else weights.with_name("config.json")
        if config_path.is_file():
            with config_path.open(encoding="utf-8") as f:
                model_config = json.load(f)
            if self.name == "uni" and model_config.get("architecture") != "vit_large_patch16_224":
                raise ValueError("UNI config architecture must be vit_large_patch16_224")
        elif config.get("config"):
            raise FileNotFoundError(f"UNI config file not found: {config_path}")
        self.device = torch.device(config["device"])
        self.batch_size = int(config["batch_size"])
        if self.batch_size < 1:
            raise ValueError("encoder.batch_size must be positive")
        self.precision = config["precision"]
        self.token_precision = config["token_precision"]
        if self.precision == "float16" and self.device.type != "cuda":
            raise ValueError("float16 encoder computation requires CUDA")
        self.normalize = bool(config["normalize"])
        self.weights_sha256 = sha256(weights)
        self.config_path = str(config_path) if config_path.is_file() else None
        kwargs = dict(pretrained=False, dynamic_img_size=True, num_classes=0)
        if self.name == "uni":
            architecture = "vit_large_patch16_224"
            kwargs.update(init_values=1e-5)
        elif self.name == "uni2_h":
            # Same architecture/kwargs as DUAL/models/uni.py and cached official config.
            architecture = "vit_giant_patch14_224"
            kwargs.update(img_size=224, patch_size=14, depth=24, num_heads=24,
                          init_values=1e-5, embed_dim=1536, mlp_ratio=2.66667 * 2,
                          no_embed_class=True, mlp_layer=timm.layers.SwiGLUPacked,
                          act_layer=torch.nn.SiLU, reg_tokens=8)
        else:
            # Exact timm architecture used by the cached Kaiko hubconf.vitb8.
            architecture = "vit_base_patch8_224"
        self.model = timm.create_model(architecture, **kwargs)
        state = torch.load(str(weights), map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        if self.model.embed_dim != self.feature_dim or tuple(self.model.patch_embed.patch_size) != (self.patch_size,) * 2:
            raise RuntimeError("Unexpected encoder patch geometry")
        mean, std = ((.5,) * 3, (.5,) * 3) if self.name == "kaiko_vitb8" else ((.485, .456, .406), (.229, .224, .225))
        self.input_normalization = {"mean": list(mean), "std": list(std), "range": "RGB uint8 / 255"}
        self.mean = torch.tensor(mean, device=self.device)[None, :, None, None]
        self.std = torch.tensor(std, device=self.device)[None, :, None, None]
        self.token_grid = None
        self.checkpoint_fingerprint = self.weights_sha256
        self.metadata = {"encoder_name": self.name, "architecture": architecture,
                         "checkpoint_path": str(weights), "checkpoint_sha256": self.weights_sha256,
                         "patch_size": self.patch_size, "feature_dim": self.feature_dim,
                         "preprocessing": self.input_normalization,
                         "feature_layer": "forward_features: final norm, spatial patches only",
                         "prefix_tokens": int(self.model.num_prefix_tokens),
                         "register_tokens": int(getattr(self.model, "num_reg_tokens", 0)),
                         "parameters": sum(p.numel() for p in self.model.parameters()),
                         "precision": self.precision, "token_precision": self.token_precision,
                         "normalize": self.normalize, "timm_version": timm.__version__}
        if self.name != "uni":
            self.metadata["batch_size"] = self.batch_size

    @torch.inference_mode()
    def extract_features(self, images):
        """RGB uint8 crops -> L2 normalized [B,H_tokens,W_tokens,C] numpy array."""
        values = self.encode(images)
        return values.reshape(len(images), *self.token_grid, self.feature_dim)

    @torch.inference_mode()
    def encode(self, crops):
        """Return [windows,spatial_tokens,C], preserving the N3 cache precision."""
        results = []
        for first in range(0, len(crops), self.batch_size):
            batch = crops[first:first + self.batch_size]
            if len(batch) == 1:
                # Preserve Direct's single-window NCHW tensor layout.
                x = torch.from_numpy(batch[0].copy()).permute(2, 0, 1)[None].to(self.device).float() / 255
            else:
                array = __import__("numpy").stack(batch)
                x = torch.from_numpy(array.copy()).permute(0, 3, 1, 2).to(self.device).float() / 255
            x = (x - self.mean) / self.std
            amp = torch.autocast(device_type="cuda", dtype=torch.float16) if self.precision == "float16" else nullcontext()
            with amp:
                features = self.model.forward_features(x)
            if features.ndim != 3:
                raise RuntimeError("Encoder did not return dense tokens")
            tokens = features[:, int(self.model.num_prefix_tokens):]
            grid = x.shape[-1] // self.patch_size
            if tokens.shape[1:] != (grid * grid, self.feature_dim):
                raise RuntimeError(f"Encoder positional embedding incompatible with {x.shape[-1]} input")
            self.token_grid = (grid, grid)
            if not torch.isfinite(tokens).all():
                raise RuntimeError("Encoder produced non-finite tokens")
            if self.normalize:
                tokens = F.normalize(tokens, dim=-1)
            tokens = tokens.half() if self.token_precision == "float16" else tokens.float()
            results.append(tokens.cpu().numpy())
        return __import__("numpy").concatenate(results, axis=0)
