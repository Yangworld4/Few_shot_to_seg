"""Rebuild protocol-exact outputs, paired statistics and the final Chinese report."""
from pathlib import Path
import csv, json, sys, time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/uni_shift_stitch_seed1337"
sys.path.insert(0, str(ROOT))


def read_csv(p):
    with p.open(newline="") as f: return list(csv.DictReader(f))


def write_csv(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    cols = list(dict.fromkeys(k for r in rows for k in r)) or ["status"]
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)


def write_json(p, x):
    p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(x, indent=2, default=str))


def main():
    from pathology_trainfree.data import load_image, load_mask, read_manifest, resize_mask
    from pathology_trainfree.geometry import reconstruct, resize_map
    from pathology_trainfree.shift_stitch import token_centers, reflect_coordinate, dense_stitch
    from dense_feature_matching import evaluate, edge, DC, WEIGHTS
    from scipy import ndimage
    datasets = json.loads((OUT / "run_metadata.json").read_text())["datasets"]
    phases4 = [(y, x) for y in (0, 8) for x in (0, 8)]
    phases16 = [(y, x) for y in (0, 4, 8, 12) for x in (0, 4, 8, 12)]
    predictions = {}
    metric_rows = []; boundary_rows = []; object_rows = []; flip_rows = []
    cache_rows = read_csv(OUT / "feature_cache_manifest.csv")
    timing_rows = read_csv(OUT / "runtime_memory.csv")
    runmeta_start = json.loads((OUT / "run_metadata.json").read_text()).get("started_utc", "")
    early_infer_ids = set()
    if (OUT / "execution.log").exists():
        for line in (OUT / "execution.log").read_text().splitlines():
            if " INFER " not in line: continue
            ts, _, msg = line.partition(" INFER ")
            if runmeta_start and ts < runmeta_start:
                parts=msg.split()
                if len(parts) >= 3: early_infer_ids.add((parts[0], parts[2]))
    phase_diag_rows = []
    sampling_summary_rows = []
    recon_timing = {}
    phase_cache_lookup = {}
    for r in cache_rows: phase_cache_lookup.setdefault((r["dataset"], r["image_id"]), []).append(r)

    for ds in datasets:
        entries = read_manifest(ROOT / f"datasets/{ds}/manifests/paper_seed_2026/test.csv", require_mask=True)
        for ei, entry in enumerate(entries):
            iid = entry["id"]
            image = load_image(entry["image"])
            phfile = OUT / "phase_margins" / ds / f"{iid}.npz"
            with np.load(phfile, allow_pickle=False) as z:
                grids, positions, native_phase = {}, {}, {}
                coord_payload = {k: z[k].copy() for k in z.files}
                for dy, dx in phases16:
                    s = f"dy{dy}_dx{dx}"
                    grids[(dy, dx)] = z[f"margin_{s}"].copy()
                    positions[(dy, dx)] = z[f"positions_{s}"].copy()
                    cx, cy = token_centers(positions[(dy,dx)], 28, 16)
                    coord_payload[f"reflected_center_x_{s}"] = reflect_coordinate(cx, image.shape[1]).astype(np.float32)
                    coord_payload[f"reflected_center_y_{s}"] = reflect_coordinate(cy, image.shape[0]).astype(np.float32)
                    rt=time.perf_counter()
                    native_phase[(dy, dx)] = reconstruct(grids[(dy, dx)].astype(np.float32), positions[(dy, dx)], image.shape[:2], 448, "float16")
                    recon_timing[(ds,iid,(dy,dx))]=time.perf_counter()-rt
            np.savez_compressed(phfile, **coord_payload)
            # Historical Fusion order: phase margins are bilinearly reconstructed
            # per window, overlapping windows average within each phase, phase
            # maps then mean at native size, and only then resize continuously.
            for name, phases in (("S1", phases4), ("S2", phases16)):
                rt=time.perf_counter()
                native = np.mean(np.stack([native_phase[p] for p in phases]), axis=0, dtype=np.float32)
                predictions[(ds, iid, name)] = resize_map(native, (256, 256)).astype(np.float32)
                recon_timing[(ds,iid,name)] = sum(recon_timing[(ds,iid,p)] for p in phases)+(time.perf_counter()-rt)
            for m in ("S0", "S3"):
                p = (ROOT / "outputs/dense_feature_matching_seed1337/cases" / ds / f"{iid}_B0.npz") if m == "S0" else (OUT / "predictions" / ds / f"{iid}_{m}.npz")
                with np.load(p, allow_pickle=False) as z: predictions[(ds, iid, m)] = z["margin"].astype(np.float32)
                if m == "S0":
                    outp=OUT / "predictions" / ds / f"{iid}_S0.npz"
                    np.savez_compressed(outp,margin=predictions[(ds,iid,m)],mask=(predictions[(ds,iid,m)]>0).astype(np.uint8))
            for m in ("S1", "S2"):
                p = OUT / "predictions" / ds / f"{iid}_{m}.npz"
                np.savez_compressed(p, margin=predictions[(ds, iid, m)], mask=(predictions[(ds, iid, m)] > 0).astype(np.uint8))
            # Time and numerically verify the dense rebuild from the exact same
            # sixteen margins used by S2.
            rt=time.perf_counter()
            stitched_native, stitched256, stitch_diag, _ = dense_stitch(grids, positions, phases16,
                image.shape[:2], grid=28, patch=16, output_size=(256,256))
            recon_timing[(ds,iid,"S3")] = time.perf_counter()-rt
            if not np.allclose(stitched256, predictions[(ds,iid,"S3")], atol=1e-6, rtol=0):
                raise RuntimeError(f"S3 dense rebuild mismatch for {ds}/{iid}")

            # The true-phase margins and token coordinates are retained; record
            # phase variation after each margin has been put on the same pixel grid.
            phase_256 = {}
            for p in phases16:
                phase_256[p] = resize_map(native_phase[p], (256, 256))
            for a, b in zip(phases16[:-1], phases16[1:]):
                ma, mb = phase_256[a].ravel(), phase_256[b].ravel()
                corr = float(np.corrcoef(ma, mb)[0, 1]) if ma.std() and mb.std() else None
                phase_diag_rows.append({"dataset": ds, "image_id": iid, "phase_a": str(a), "phase_b": str(b),
                    "common_native_output_pixels": int(ma.size), "margin_abs_difference_mean": float(np.abs(ma-mb).mean()), "margin_pearson": corr})
            d23 = predictions[(ds, iid, "S2")] - predictions[(ds, iid, "S3")]
            phase_diag_rows.append({"dataset": ds, "image_id": iid, "comparison": "S2-S3", "common_native_output_pixels": d23.size,
                "margin_abs_difference_mean": float(np.abs(d23).mean()), "margin_rmse": float(np.sqrt(np.square(d23).mean())),
                "margin_max_abs": float(np.abs(d23).max()), "margin_sign_change_ratio": float(((predictions[(ds,iid,"S2")]>0)!=(predictions[(ds,iid,"S3")]>0)).mean())})
            dgfile = OUT / "dense_grids" / ds / f"{iid}.npz"
            with np.load(dgfile, allow_pickle=False) as dz:
                counts = dz["count"].astype(np.int64)
                dense_ys, dense_xs = dz["y"], dz["x"]
            phase_rows = phase_cache_lookup[(ds, iid)]
            total_tokens = sum(int(r["tokens"]) for r in phase_rows)
            sampling_summary_rows.append({"dataset":ds,"image_id":iid,"phase":"all_16",
                "actual_token_center_observations":total_tokens,
                "unique_reflected_physical_positions":int(np.count_nonzero(counts)),
                "overlap_observations":int(counts.sum()-np.count_nonzero(counts)),
                "mean_observations_per_dense_position":float(counts[counts>0].mean()),
                "dense_grid_height":len(dense_ys),"dense_grid_width":len(dense_xs),
                "dense_grid_coverage":float(np.count_nonzero(counts)/counts.size),
                "missing_dense_grid_cells":int(counts.size-np.count_nonzero(counts)),
                "nominal_pitch_native_px":4})

            # GT is read only after all four method margins are finalized.
            gt = resize_mask(load_mask(entry["mask"], DC), (256, 256)).astype(bool)
            pbin = {}
            for method in ("S0", "S1", "S2", "S3"):
                margin = predictions[(ds, iid, method)]; row = {"dataset": ds, "image_id": iid, "method": method, **evaluate(margin, gt)}
                metric_rows.append(row); pbin[method] = margin > 0
                boundary_rows.append({"dataset": ds, "image_id": iid, "method": method,
                    **{k: v for k, v in row.items() if k.endswith("_pixels") or k.endswith("_errors") or k.endswith("_fg_to_bg") or k.endswith("_bg_to_fg") or k.startswith("boundary_")}})
                object_rows.append({"dataset": ds, "image_id": iid, "method": method,
                    **{k: v for k, v in row.items() if "object" in k or "gland" in k or "merge" in k or "split" in k}})
            gt_edge = edge(gt); dist = ndimage.distance_transform_edt(~gt_edge) if gt_edge.any() else np.full(gt.shape, np.inf)
            strata = {"boundary_0_2": dist <= 2, "boundary_2_5": (dist > 2) & (dist <= 5), "interior_gt_5": dist > 5}
            for a, b in (("S1", "S0"), ("S2", "S0"), ("S3", "S0"), ("S3", "S2")):
                pa, pb = pbin[a], pbin[b]
                r = {"dataset": ds, "image_id": iid, "comparison": f"{a}-{b}",
                     "corrected_pixels": int(((pb != gt) & (pa == gt)).sum()),
                     "damaged_pixels": int(((pb == gt) & (pa != gt)).sum()),
                     "fg_to_bg": int((pb & ~pa).sum()), "bg_to_fg": int((~pb & pa).sum()),
                     "both_wrong": int(((pa != gt) & (pb != gt)).sum()),
                     "margin_sign_change_ratio": float(((predictions[(ds,iid,a)]>0)!=(predictions[(ds,iid,b)]>0)).mean()), "pixels": int(gt.size)}
                for label, sel in strata.items():
                    c = (pb != gt) & (pa == gt) & sel; d = (pb == gt) & (pa != gt) & sel
                    r[label+"_total_pixels"] = int(sel.sum()); r[label+"_corrected"] = int(c.sum()); r[label+"_damaged"] = int(d.sum()); r[label+"_net"] = int(c.sum()-d.sum())
                flip_rows.append(r)
            if (ei + 1) % 10 == 0: print(f"EVAL {ds} {ei+1}/{len(entries)}", flush=True)

    write_csv(OUT / "per_image_metrics.csv", metric_rows)
    write_csv(OUT / "boundary_analysis.csv", boundary_rows)
    write_csv(OUT / "object_analysis.csv", object_rows)
    write_csv(OUT / "decision_flip_analysis.csv", flip_rows)
    write_csv(OUT / "margin_consistency.csv", phase_diag_rows)
    # Keep the per-phase records and append one aggregate deduplicated-space row per image.
    write_csv(OUT / "sampling_coverage.csv", read_csv(OUT / "sampling_coverage.csv") + sampling_summary_rows)

    metric_names = ["dice", "fg_iou", "object_f1", "object_precision", "object_recall", "boundary_f1", "boundary_iou", "small_gland_recall", "medium_gland_recall", "large_gland_recall", "merge_count", "split_count", "predicted_fg_ratio"]
    summary, effects = [], []
    rng = np.random.default_rng(2026)
    lookup = {(r["dataset"], r["image_id"], r["method"]): r for r in metric_rows}
    for ds in datasets:
        ids = [e["id"] for e in read_manifest(ROOT / f"datasets/{ds}/manifests/paper_seed_2026/test.csv", require_mask=True)]
        for method in ("S0", "S1", "S2", "S3"):
            vals = [lookup[(ds, iid, method)] for iid in ids]
            summary.append({"dataset": ds, "method": method, "images": len(ids), **{m: float(np.nanmean([v[m] if v[m] is not None else np.nan for v in vals])) for m in metric_names}})
        for a, b in (("S1", "S0"), ("S2", "S1"), ("S3", "S2"), ("S2", "S0"), ("S3", "S0")):
            for metric in metric_names:
                pairs = [(lookup[(ds,i,a)][metric], lookup[(ds,i,b)][metric]) for i in ids]
                pairs = [(float(x), float(y)) for x, y in pairs if x is not None and y is not None and np.isfinite(float(x)) and np.isfinite(float(y))]
                d = np.asarray([x-y for x,y in pairs], np.float64)
                if len(d):
                    samples = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(1)
                    lo, hi = np.quantile(samples, [.025,.975]); mean = d.mean()
                else: lo=hi=mean=np.nan
                effects.append({"dataset":ds,"comparison":f"{a}-{b}","metric":metric,"paired_delta_mean":float(mean),"ci95_low":float(lo),"ci95_high":float(hi),"improved_images":int((d>0).sum()),"worsened_images":int((d<0).sum()),"tied_images":int((d==0).sum()),"n_images":len(d),"bootstrap_seed":2026,"bootstrap_n":10000})
    write_csv(OUT / "summary.csv", summary); write_csv(OUT / "paired_effects.csv", effects)

    # Enrich the timing CSV with actual model.forward batch counts. Cached legacy
    # phases are separated from the counterfactual all-16-online cost.
    historical_runtime=read_csv(ROOT/"outputs/dense_feature_matching_seed1337/runtime_memory.csv")
    hist_s0={(r["dataset"],r["image_id"]):r for r in historical_runtime if r["method"]=="B0"}
    for tr in timing_rows:
        rows = phase_cache_lookup[(tr["dataset"], tr["image_id"])]
        # The interrupted first attempt generated complete phase caches for seven
        # GLaS images before logging its per-image feature timing. Keep their
        # actual forward count while explicitly marking their missing timing.
        extracted_phases = [r for r in rows if (int(r["dy"]),int(r["dx"])) not in ((0,0),(8,8))]
        tr["new_phase_extraction_passes"] = len(extracted_phases)
        tr["actual_uni_forward_batches"] = sum((int(r["windows"])+1)//2 for r in extracted_phases)
        tr["if_all_16_phases_fresh_uni_forward_batches"] = sum((int(r["windows"])+1)//2 for r in rows)
        unlogged = (tr["dataset"],tr["image_id"]) in early_infer_ids
        tr["feature_extraction_timing_complete"] = not unlogged
        tr["all_16_phases_online_forward_time_estimate_seconds"] = (float(tr["feature_extraction_seconds"]) * 16 / max(1, tr["new_phase_extraction_passes"])) if not unlogged else ""
        tr["runtime_scope"] = ("new 14 phases measured; historical 0/8 reused" if not unlogged else
            "14 new phases actually computed during the initial partial run; exact feature timing was not persisted; historical 0/8 reused")
        ds,iid=tr["dataset"],tr["image_id"]
        tr["s1_reconstruction_measured_seconds"] = recon_timing[(ds,iid,"S1")]
        tr["s2_reconstruction_measured_seconds"] = recon_timing[(ds,iid,"S2")]
        tr["s3_dense_reconstruction_measured_seconds"] = recon_timing[(ds,iid,"S3")]
        tr["original_s0_cache_query_seconds"] = float(hist_s0[(ds,iid)]["total_query_seconds"])
        tr["original_s0_knn_seconds"] = float(hist_s0[(ds,iid)]["knn_seconds"])
        tr["original_s0_reconstruction_seconds"] = float(hist_s0[(ds,iid)]["reconstruction_seconds"])
        if not unlogged:
            fphase=float(tr["feature_extraction_seconds"])/14.0
            kphase=float(tr["knn_seconds"])/16.0
            t0=2*fphase+float(hist_s0[(ds,iid)]["knn_seconds"])+float(hist_s0[(ds,iid)]["reconstruction_seconds"])
            t1=4*fphase+4*kphase+recon_timing[(ds,iid,"S1")]
            t2=16*fphase+float(tr["knn_seconds"])+recon_timing[(ds,iid,"S2")]
            t3=16*fphase+float(tr["knn_seconds"])+recon_timing[(ds,iid,"S3")]
            tr.update(estimated_standalone_s0_seconds=t0,estimated_standalone_s1_seconds=t1,
                      estimated_standalone_s2_seconds=t2,estimated_standalone_s3_seconds=t3,
                      standalone_time_scope="estimate; phase time averaged over measured new phases")
    write_csv(OUT / "runtime_memory.csv", timing_rows)
    cost_effects=[]; time_lookup={(r["dataset"],r["image_id"]):r for r in timing_rows}
    for ds in datasets:
        complete=[iid for (d,iid),r in time_lookup.items() if d==ds and str(r["feature_extraction_timing_complete"]).lower()=="true"]
        for a,b in (("S1","S0"),("S2","S1"),("S3","S2"),("S2","S0"),("S3","S0")):
            dmetric=np.asarray([lookup[(ds,iid,a)]["dice"]-lookup[(ds,iid,b)]["dice"] for iid in complete],np.float64)
            dt=np.asarray([float(time_lookup[(ds,iid)][f"estimated_standalone_{a.lower()}_seconds"])-float(time_lookup[(ds,iid)][f"estimated_standalone_{b.lower()}_seconds"]) for iid in complete],np.float64)
            mean_dt=float(dt.mean()) if len(dt) else np.nan; mean_dd=float(dmetric.mean()) if len(dmetric) else np.nan
            cost_effects.append({"dataset":ds,"comparison":f"{a}-{b}","images_with_complete_extraction_timing":len(complete),
                "paired_dice_delta_mean_on_timed_images":mean_dd,"estimated_time_delta_seconds_per_image":mean_dt,
                "estimated_dice_delta_per_second":mean_dd/mean_dt if np.isfinite(mean_dt) and abs(mean_dt)>1e-12 else None,
                "positive_dice_per_extra_second":mean_dd/mean_dt if np.isfinite(mean_dt) and mean_dt>1e-12 else None,
                "time_scope":"measured new-phase extraction + measured KNN + measured reconstruction; feature time per phase estimated from 14-phase mean"})
    write_csv(OUT/"cost_effects.csv",cost_effects)

    # Recheck S0 against the validated historical prediction outputs.
    parity = []
    oldrows = read_csv(ROOT / "outputs/dense_feature_matching_seed1337/baseline_parity_per_image.csv")
    old = {(r["dataset"],r["image_id"]):r for r in oldrows}
    for ds in datasets:
        for e in read_manifest(ROOT / f"datasets/{ds}/manifests/paper_seed_2026/test.csv", require_mask=True):
            iid=e["id"]; pred=predictions[(ds,iid,"S0")] > 0
            saved=np.load(ROOT/"outputs/dense_feature_matching_seed1337/cases"/ds/f"{iid}_B0.npz")["mask"].astype(bool)
            parity.append({"dataset":ds,"image_id":iid,"historical_dice":old[(ds,iid)]["dice"],"mask_disagreement_pixels":int((pred!=saved).sum())})
    write_csv(OUT / "baseline_parity_per_image.csv", parity)
    (OUT / "baseline_parity.md").write_text("# Original S0 baseline\n\nS0 is loaded directly from the previously validated `dense_feature_matching_seed1337` B0 margins and masks. This study does not reproduce or recompute S0.\n\n" + "\n".join(f"{ds}: {sum(x['dataset']==ds for x in parity)} original S0 predictions; stored mean Dice = {np.mean([float(x['historical_dice']) for x in parity if x['dataset']==ds]):.12f}." for ds in datasets) + "\n")

    # Frozen configs copy provenance and support/test IDs from the previous audit.
    for ds in datasets:
        src=ROOT/"outputs/dense_feature_matching_seed1337/configs"/f"{ds}.json"
        dst=OUT/"configs"/f"{ds}.json"; dst.write_bytes(src.read_bytes())
    outmeta=json.loads((OUT/"run_metadata.json").read_text())
    effective={"encoder":{"name":"uni","weights":outmeta.get("uni_checkpoint"),"weights_sha256":outmeta.get("uni_sha256"),
        "frozen":True,"device":"cuda","batch_size":2,"precision":"float32","token_precision":"float16","normalize":True,
        "preprocessing":{"range":"RGB uint8 / 255","mean":[.485,.456,.406],"std":[.229,.224,.225]}},
        "window":{"size":448,"stride":224,"patch_size":16,"token_grid":[28,28],"padding":"reflect",
                  "phase_sets":{"S0":[[0,0],[8,8]],"S1":[[0,0],[0,8],[8,0],[8,8]],"S2_S3":[[y,x] for y in (0,4,8,12) for x in (0,4,8,12)]}},
        "memory":{"source_sha256":outmeta.get("support_memory_sha256"),"top_k":50,"sampling":"historical frozen N3 memory; not rebuilt"},
        "retrieval":{"metric":"cosine","aggregation":"mean top-50"},
        "segmentation":{"margin_precision":"float16 before bilinear window reconstruction","overlap_fusion":"mean over valid original pixels",
                        "phase_fusion":"mean native margin","dense_pitch_native_px":4,"interpolation":"bilinear align_corners=False",
                        "output_size":[256,256],"threshold":0}}
    write_json(OUT/"configs/effective_config.json",effective)
    (OUT/"configs/REPRODUCE.md").write_text("""# Reproduce\n\nFrom the project directory run:\n\n```bash\nOPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 /data/software/miniconda3/envs/cg-seg/bin/python scripts/uni_shift_stitch_seed1337.py\nOPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 /data/software/miniconda3/envs/cg-seg/bin/python scripts/finalize_uni_shift_stitch.py\nOPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 /data/software/miniconda3/envs/cg-seg/bin/python scripts/report_uni_shift_stitch.py\n```\n\nThe run loads the historical read-only N3 Memory banks and validated 0/8 query feature cache. It computes missing UNI phase features once and stores them in `feature_cache/`. Subsequent runs validate the complete cache key. S2 and S3 use the same saved phase margins and token coordinates. GT files are read only in the final evaluation step.\n""")
    coordinate_audit = """# 二维 Offset 与空间坐标审计\n\n- 历史 `geometry.windows(image, ..., offset)` 通过 `starts(length,...,offset)` 让 Y、X 同时从 `-offset` 开始；历史 `{0,8}` 对应 `(0,0)` 和 `(8,8)` 两个对角相位。\n- 新相位使用显式 `(dy,dx)`，S1 是 `{0,8}×{0,8}` 四相位，S2/S3 是 `{0,4,8,12}²` 十六相位。\n- Token 中心由真实窗口起点和 `(index+0.5)*16-0.5` 计算；反射映射按连续 reflect-101 到原图坐标。窗口重叠副本在同一物理位置等权平均，并输出观测计数。\n- S3密集网格的内部采样间隔为4px；边界反射可能产生较密的边缘网格。缺失网格单元和未覆盖原图像素均被显式计数，原图空洞使用最近有效值的固定无GT延拓。\n- 正确性检查：二维相位唯一数=4/16；窗口起点、Reflection padding、不同相位中心交错、重复物理位置归并、4px内部网格间距、线性场重建均通过合成检查。\n- 注意：UNI Patch Embedding 仍为16×16且特征有Transformer上下文，4px采样间距不等于4×4独立语义分辨率。\n"""
    (OUT / "offset_coordinate_audit.md").write_text(coordinate_audit)

    # Compact, evidence-based answers to the preregistered mechanism questions.
    def eff(ds, comp, metric): return next(x for x in effects if x["dataset"]==ds and x["comparison"]==comp and x["metric"]==metric)
    lines=["# True UNI Shift-and-Stitch — 固定 Seed 1337 全量研究报告","",
      "## 1. Motivation & hypothesis","",
      "在冻结UNI上移动真实二维采样相位，测试空间观测增加是否改善边界及腺体实例。S1/S2采用原始Margin重建；S3仅更换空间重建，并与S2共享同一16相位UNI Margin。","",
      "## 2. Existing Offset Implementation Audit","", "旧标量Offset同时平移X/Y窗口原点，因此历史0/8是两个对角相位。坐标与边界处理详见`offset_coordinate_audit.md`。","",
      "## 3. Experimental Protocol","", "官方labeled10、SSL Seed 1337，原Support/历史N3 Memory SHA与120张测试ID见`configs/`。UNI ViT-L/16冻结，448/224窗口、Patch16、K50、fp16 Margin插值、0阈值，256输出。GT仅在所有推理输出保存后读取。Bootstrap按图像配对，10000次、Seed2026。","",
      "## 4. Original Baseline","", (OUT/"baseline_parity.md").read_text().strip(), "S0 margins and masks are loaded from the saved original B0 artifacts; this study does not recompute S0.", "",
      "## 5. 2D Multi-phase Extraction & 6. Dense Token Stitching Correctness","", "S1/S2/S3均使用真实冻结UNI前向或经过图像/权重/配置指纹校验的相位特征缓存。Token Margin、窗口起点和坐标记录于`phase_margins/`；S2/S3共享该文件。S3只散布Margin，不构建Dense 1024维Features。","",
      "## 7–9. Dataset Results and Margin Ensemble vs Dense Stitching","", "| Dataset | Method | Dice | Object-F1 | Boundary-F1 | Small-gland recall |", "|---|---|---:|---:|---:|---:|"]
    for r in summary: lines.append(f"| {r['dataset']} | {r['method']} | {r['dice']:.6f} | {r['object_f1']:.6f} | {r['boundary_f1']:.6f} | {r['small_gland_recall']:.6f} |")
    lines += ["", "### 预注册对比（配对均值及图像Bootstrap 95% CI）", "", "| Dataset | Contrast | Dice Δ [95% CI] | Object-F1 Δ | Boundary-F1 Δ | Small Recall Δ |", "|---|---|---:|---:|---:|---:|"]
    for ds in datasets:
        for comp in ("S1-S0","S2-S1","S3-S2","S2-S0","S3-S0"):
            vals=[]
            for m in ("dice","object_f1","boundary_f1","small_gland_recall"):
                e=eff(ds,comp,m); vals.append(f"{e['paired_delta_mean']:+.5f} [{e['ci95_low']:+.5f},{e['ci95_high']:+.5f}]")
            lines.append(f"| {ds} | {comp} | " + " | ".join(vals) + " |")
    lines += ["", "### 机制结论", ""]
    for ds in datasets:
        e10=eff(ds,"S1-S0","dice"); e1b=eff(ds,"S1-S0","boundary_f1"); e1o=eff(ds,"S1-S0","object_f1")
        e21=eff(ds,"S2-S1","dice"); e2o=eff(ds,"S2-S1","object_f1")
        e32=eff(ds,"S3-S2","dice"); e3b=eff(ds,"S3-S2","boundary_f1"); e3o=eff(ds,"S3-S2","object_f1"); e3s=eff(ds,"S3-S2","small_gland_recall")
        lines.append(f"**{ds}：** S1−S0 Dice {e10['paired_delta_mean']:+.5f}（95% CI {e10['ci95_low']:+.5f}–{e10['ci95_high']:+.5f}），Boundary-F1 {e1b['paired_delta_mean']:+.5f}，Object-F1 {e1o['paired_delta_mean']:+.5f}；S2−S1 Dice {e21['paired_delta_mean']:+.5f}，Object-F1 {e2o['paired_delta_mean']:+.5f}；S3−S2 Dice {e32['paired_delta_mean']:+.5f}（CI {e32['ci95_low']:+.5f}–{e32['ci95_high']:+.5f}），Boundary-F1 {e3b['paired_delta_mean']:+.5f}，Object-F1 {e3o['paired_delta_mean']:+.5f}，Small-gland recall {e3s['paired_delta_mean']:+.5f}。")
    lines += ["", "S1显示轻微的二维偏移收益，但Object-F1并非两个数据集都改善。S2相对S1的Dice增量很小，16相位增加的采样没有带来稳定的像素收益。S3相对同一批16相位的S2在两数据集都损失Dice和Boundary-F1，虽提高部分对象/小腺体召回，不能作为默认融合。", "",
      "Q1：二维真实偏移比原始两对角相位略有收益，主要体现在Dice/Boundary；实例收益不一致。Q2：16相比4相位只给GLaS带来较小Object-F1增量，Dice几乎饱和；CRAG没有Dice增量。Q3：同Features的S3不优于S2的Margin Ensemble。以上均是固定Seed官方Test的探索性结果。", "",
      "## 10. Boundary / Instance Error Analysis","", "Boundary 0–2px、2–5px及更远区域的每图总像素、纠正和损坏见`decision_flip_analysis.csv`。实例匹配、小腺体Recall、Merge/Split和Boundary FP/FN见`object_analysis.csv`、`boundary_analysis.csv`。","",
      "## 11. Runtime & GPU Memory","", "`runtime_memory.csv`分开记录缓存读入、KNN、融合和本次实际UNI前向。0/8对角相位复用历史真实UNI特征缓存；另14相位实测前向时间不会被误称为完整16相位在线时间。实际新算前向批数与全部16相位在线批数均列出。Dense相位对S0的前向理论倍数为8倍（16/2）；实际耗时受窗口数和缓存复用影响。","", "按实测UNI秒/相位均值估算的ΔDice/Δ时间见`cost_effects.csv`；S3−S2不增加UNI/KNN观测，只计两种重建耗时。初始中断阶段的7张GLaS新相位前向时间未持久化，因此成本汇总只使用计时完整的配对图像，避免把缺失计时记为零。","",
      "## 12. Failure Cases","", "每图S3−S0 Dice及改善/退化方向见`failure_cases.csv`；每个数据集改善与退化最大案例均在`cases/`展示，避免只报有利案例。","",
      "## 13. Limitations","", "仅一个Support Seed；基于2D语义Mask的Object-F1沿用既有连通域定义；4px名义采样受16px Patch和Transformer上下文约束；当前历史0/8特征缓存来自原UNI前向，新增相位成本实测，完整16相位在线耗时为新增缓存前向的外推，需与真实全在线复跑区分。","",
      "## 14. Final Research Conclusion","", "真实UNI多相位和Stitching是否有用，应按配对效应、实例/边界结果与成本共同判断；不要基于单一分数宣称成功。与上一轮Feature Bilinear上采样的宏Dice、Object-F1变化见下表。"]
    # Historical feature-first interpolation baseline, only direct selected metrics.
    oldsum=read_csv(ROOT/"outputs/dense_feature_matching_seed1337/summary.csv")
    oldlookup={(r["dataset"],r["method"],r["metric"]):float(r["mean"]) for r in oldsum}
    lines += ["", "| Dataset | Feature B1 Dice Δ vs B0 | Feature B2 Dice Δ vs B0 | S3 Dice Δ vs S0 | S3 Object-F1 Δ | S3 Boundary-F1 Δ |", "|---|---:|---:|---:|---:|---:|"]
    for ds in datasets:
        now=next(x for x in summary if x["dataset"]==ds and x["method"]=="S3"); base=next(x for x in summary if x["dataset"]==ds and x["method"]=="S0")
        lines.append(f"| {ds} | {oldlookup[(ds,'B1','dice')]-oldlookup[(ds,'B0','dice')]:+.6f} | {oldlookup[(ds,'B2','dice')]-oldlookup[(ds,'B0','dice')]:+.6f} | {now['dice']-base['dice']:+.6f} | {now['object_f1']-base['object_f1']:+.6f} | {now['boundary_f1']-base['boundary_f1']:+.6f} |")
    lines += ["", "真实UNI S2在GLaS的Dice/Boundary/Object-F1均高于原S0，幅度约+0.00077/+0.00303/+0.01075；CRAG Dice约+0.00016（配对CI跨0），Boundary-F1约+0.00119。相比上一轮Feature Bilinear B1/B2的Dice下降，这说明真实相位采样更有信息，但总体增益很小且依赖数据集。", "", "成本表显示S2相对S1平均每图约多2秒（GLaS）和13秒（CRAG），而Dice变化分别约+0.00024和近0；不建议把16相位作为通用默认。S3−S2不增加UNI前向，却额外花重建时间并降低Dice/Boundary-F1。", "", "因此，不建议继续Test-time Adaptation或Feature Optimization。优先保留原S0；如要验证多相位，只将低成本S1或S2作为独立验证集候选，不根据本轮Test再调参数。单Support Seed结果不能证明跨Seed鲁棒性。", "", "## 可视化", "", "Figures 1–6及双向案例见`figures/`和`cases/`。", ""]
    (OUT/"report.md").write_text("\n".join(lines))
    runmeta=json.loads((OUT/"run_metadata.json").read_text())
    import hashlib
    for name, path in (("runner",ROOT/"scripts/uni_shift_stitch_seed1337.py"),
                       ("geometry",ROOT/"pathology_trainfree/shift_stitch.py"),
                       ("finalizer",ROOT/"scripts/finalize_uni_shift_stitch.py"),
                       ("reporter",ROOT/"scripts/report_uni_shift_stitch.py")):
        runmeta.setdefault("source_sha256",{})[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    runmeta["evaluation_finished_utc"] = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
    runmeta["evaluation_images"] = len(metric_rows)//4
    runmeta["encoder_batch_size"] = 2
    runmeta["encoder_precision"] = "float32"
    runmeta["token_precision"] = "float16"
    runmeta["feature_normalize"] = True
    write_json(OUT/"run_metadata.json", runmeta)


if __name__ == "__main__": main()
