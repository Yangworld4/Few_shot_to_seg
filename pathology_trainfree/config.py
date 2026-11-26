"""Load experiment presets and their shared YAML defaults."""

from pathlib import Path

import yaml


def read_config(path, ancestors=()):
    path = Path(path).resolve()
    if path in ancestors:
        chain = " -> ".join(str(p) for p in (*ancestors, path))
        raise ValueError(f"Circular config defaults: {chain}")
    with path.open(encoding="utf-8") as stream:
        preset = yaml.safe_load(stream)
    if not isinstance(preset, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {path}")
    defaults = preset.pop("defaults", [])
    if not isinstance(defaults, list) or any(not isinstance(base, str) or not base for base in defaults):
        raise ValueError(f"Config defaults must be a list of YAML paths: {path}")
    cfg = {}
    bases = [read_config(path.parent / base, (*ancestors, path)) for base in defaults]
    for layer in [*bases, preset]:
        for section, fields in layer.items():
            if isinstance(cfg.get(section), dict) and isinstance(fields, dict):
                cfg[section] = {**cfg[section], **fields}
            else:
                cfg[section] = fields
    return cfg
