"""Support retrieval and Query refinement from frozen features."""

import hashlib
import math
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import torch
import torch.nn.functional as F

from .geometry import windows
from .selfref_geometry import GRID, PATCH, STRIDE, WINDOW, phase_candidates, support_candidates


def class_budgets(counts, mode, rho, cap):
    budget = math.ceil(rho * sum(counts) / 2) if mode == "ratio" else cap
    return tuple(min(budget, count) for count in counts)


@torch.inference_mode()
def kcenter_order(features, count):
    count = min(count, len(features))
    if not count:
        return np.empty(0, dtype=np.int64)
    mean = F.normalize(features.mean(0, keepdim=True), dim=1)
    point = (features @ mean.T).argmax().reshape(1)
    order = torch.empty(count, dtype=torch.long, device=features.device)
    nearest = torch.full((len(features),), torch.inf, device=features.device, dtype=features.dtype)
    used = torch.zeros_like(nearest, dtype=torch.bool)
    for i in range(count):
        order[i:i + 1] = point
        used.scatter_(0, point, True)
        distance = (1 - (features @ features.index_select(0, point).T).flatten()).clamp_min_(0)
        nearest = torch.minimum(nearest, distance)
        point = nearest.masked_fill(used, -1).argmax().reshape(1)
    return order.cpu().numpy()


def _cached_order(candidates, count, directory):
    if directory is None:
        return kcenter_order(candidates, count)
    digest = hashlib.sha256(f"kcenter-v1:{tuple(candidates.shape)}:{candidates.device.type}:{torch.__version__}".encode())
    digest.update(candidates.cpu().contiguous().numpy().tobytes())
    path = Path(directory) / f"{digest.hexdigest()}_{count}.npy"
    if path.is_file():
        order = np.load(path, allow_pickle=False)
        if (order.shape == (count,) and order.dtype == np.int64 and len(np.unique(order)) == count
                and np.all((order >= 0) & (order < len(candidates)))):
            return order
        raise ValueError(f"Invalid k-center cache: {path}")
    order = kcenter_order(candidates, count)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
        try:
            np.save(file, order, allow_pickle=False)
            file.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return order


class SupportBank:
    def __init__(self, features, indices):
        self.features = features
        self.indices = indices
        self.sizes = tuple(len(idx) for idx in indices)


def prepare_support(features, positions, truth, cfg):
    is_n3 = cfg["feature_format"] == "n3"
    if is_n3:
        occupancy = np.concatenate([
            np.stack([crop.reshape(GRID, PATCH, GRID, PATCH).mean(axis=(1, 3)).ravel()
                      for _, _, crop in windows(truth, WINDOW, STRIDE, phase)])
            .ravel() for phase in cfg["phases"]
        ]).astype(np.float32)
        purity = (1 - occupancy, occupancy)
        rows = tuple(np.flatnonzero(p >= t) for p, t in zip(purity, (1 - cfg["purity_bg"], cfg["purity_fg"])))
        values = np.concatenate(features)
        # Historical Direct falls back to the nearest 10% by occupancy.
        fallback = max(1, math.ceil(.1 * len(occupancy)))
        if cfg["budget_mode"] == "fixed" and cfg["bank_select"] != "all":
            fallback = min(fallback, cfg["budget_cap"])
        rows = tuple(ids if len(ids) else np.argsort(-purity[c], kind="stable")[:fallback]
                     for c, ids in enumerate(rows))
    else:
        values, labels = support_candidates(features, positions, truth, cfg["min_valid_pixels"],
                                            cfg["purity_fg"], cfg["purity_bg"])
        rows = (np.flatnonzero(~labels), np.flatnonzero(labels))
    if not all(len(ids) for ids in rows):
        raise ValueError("Both Support classes need at least one candidate")
    return values, rows


@torch.inference_mode()
def build_support(values, rows, cfg):
    is_n3 = cfg["feature_format"] == "n3"
    if is_n3:
        whole = torch.as_tensor(values, device=cfg["device"], dtype=torch.float16)
    else:
        whole = F.normalize(torch.as_tensor(values, device=cfg["device"]), dim=1).half()
    budgets = class_budgets(tuple(map(len, rows)), cfg["budget_mode"], cfg["rho"], cfg["budget_cap"])
    rng = np.random.default_rng(cfg["random_seed"])
    selected = [None, None]
    # Direct's RNG samples FG before BG and sorts the selected indices.
    for c in (1, 0):
        ids, count = rows[c], budgets[c]
        if cfg["bank_select"] == "all":
            chosen = ids
        elif cfg["bank_select"] == "random":
            chosen = ids if len(ids) <= count else np.sort(rng.choice(ids, size=count, replace=False))
        else:
            candidates = F.normalize(whole[ids].float(), dim=1)
            chosen = ids[_cached_order(candidates, count, cfg["cache_dir"])]
        selected[c] = torch.as_tensor(chosen, device=whole.device)
    if is_n3:
        if any(len(ids) < cfg["support_topk"] for ids in selected):
            raise ValueError("Historical N3 retrieval requires at least support_topk candidates per class")
        compact = tuple(whole.index_select(0, ids).cpu() for ids in selected)
        return SupportBank(compact, selected)
    return SupportBank(whole.cpu(), tuple(ids.cpu() for ids in selected))


@torch.inference_mode()
def cross_image_margin(query, bank, cfg):
    k = cfg["support_topk"]
    if cfg["feature_format"] == "n3":
        banks = tuple(F.normalize(v.to(cfg["device"]).float(), dim=1) for v in bank.features)
        values = [tuple((part @ bank.T).topk(min(k, len(bank)), dim=1).values.mean(1)
                        for bank in banks) for part in query.split(GRID**2)]
        return torch.cat([fg - bg for bg, fg in values]).cpu().numpy()
    whole = bank.features.to(cfg["device"])
    bg, fg = (i.to(cfg["device"]) for i in bank.indices)
    # PyTorch 2.0's CPU backend requires float32 for the matrix multiply.
    cpu = not query.is_cuda
    if cpu:
        query, whole = query.float(), whole.float()
    margins = []
    for part in query.split(1024):
        sim = part @ whole.T
        if cpu:
            sim = sim.half().float()
        scores = [sim.index_select(1, ids).topk(min(k, len(ids)), dim=1).values.float().mean(1)
                  for ids in (bg, fg)]
        margins.append(scores[1] - scores[0])
    return torch.cat(margins).cpu().numpy()


def select_seeds(values, ratio):
    seeds = []
    for side in (values < 0, values > 0):
        ids = np.flatnonzero(side)
        ranked = ids[np.argsort(-np.abs(values[ids]), kind="stable")]
        seeds.append(np.sort(ranked[:math.ceil(ratio * len(ids))]))
    return seeds


class CachedQuery:
    @torch.inference_mode()
    def __init__(self, values, positions, shape, cfg):
        self.full = F.normalize(torch.as_tensor(np.concatenate(values), device=cfg["device"], dtype=torch.float32), dim=1)
        self.cross_query = self.full if cfg["feature_format"] == "n3" else self.full.half()
        self.similarities = None
        self.check_cache = True
        if "query_refined" not in cfg["branches"]:
            return
        chosen, keys = [], []
        start = 0
        for pos in positions:
            centers, _, _, _, phase_ids = phase_candidates(pos, shape, cfg["min_valid_pixels"])
            chosen.append(start + phase_ids)
            doubled = (centers * 2).astype(np.int64)
            keys.append(doubled[:, 1] * (2 * shape[1]) + doubled[:, 0])
            start += len(centers)
        self.chosen = np.concatenate(chosen)
        self.keys = torch.as_tensor(np.concatenate(keys), device=self.full.device)
        self.candidates = self.full[self.chosen]
        self.candidate_keys = self.keys[self.chosen]
        if self.full.is_cuda and len(self.candidates):
            free, _ = torch.cuda.mem_get_info(self.full.device)
            required = len(self.full) * len(self.candidates) * 4
            if required <= .70 * free:
                self.similarities = self.full.new_empty((len(self.full), len(self.candidates)))
                for i in range(0, len(self.full), 256):
                    self.similarities[i:i + 256] = self.full[i:i + 256] @ self.candidates.T

    @torch.inference_mode()
    def refine(self, m0, cfg):
        seed_ids = select_seeds(m0[self.chosen], cfg["seed_ratio"])
        if not all(len(ids) for ids in seed_ids):
            return m0
        ids = [torch.as_tensor(i, device=self.full.device) for i in seed_ids]
        banks = [self.candidates[i] for i in ids]
        locations = [self.candidate_keys[i] for i in ids]
        k = cfg["selfref_topk"]
        if self.similarities is not None and self.check_cache:
            ix, bank = ids[0], banks[0]
            error = (self.full[:8] @ bank.T - self.similarities[:8].index_select(1, ix)).abs().max()
            if error > 1e-6:
                self.similarities = None
            self.check_cache = False
        margins = []
        for start in range(0, len(self.full), 4096):
            x = self.full[start:start + 4096]
            keys = self.keys[start:start + len(x)]
            scores = []
            for ix, bank, seeds in zip(ids, banks, locations):
                sim = (x @ bank.T if self.similarities is None
                       else self.similarities[start:start + len(x)].index_select(1, ix))
                if cfg["exclude_same_position"]:
                    allowed = keys[:, None] != seeds[None, :]
                    sim.masked_fill_(~allowed, -torch.inf)
                    count = allowed.sum(1).clamp(max=k)
                else:
                    count = torch.full((len(x),), min(k, len(ix)), device=x.device)
                top = sim.topk(min(k, len(ix)), dim=1).values
                rank = torch.arange(top.shape[1], device=x.device)
                scores.append(top.masked_fill(rank >= count[:, None], 0).sum(1) / count)
            margins.append(scores[1] - scores[0])
        m1 = torch.cat(margins).cpu().numpy()
        return np.where(np.isfinite(m1), m1, m0)


def predict_margins(query, bank, cfg):
    support_margin = cross_image_margin(query.cross_query, bank, cfg)
    return {name: query.refine(support_margin, cfg) if name == "query_refined" else support_margin
            for name in cfg["branches"]}
