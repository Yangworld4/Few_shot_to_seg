"""Create paired summaries and figures after the fixed shift-and-stitch run."""
from pathlib import Path
import csv, json, sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/uni_shift_stitch_seed1337"
sys.path.insert(0, str(ROOT))


def read_csv(path):
    with path.open(newline="") as f: return list(csv.DictReader(f))


def fnum(x): return float(x) if x not in (None, "", "None") else np.nan


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import ndimage
    from matplotlib.colors import ListedColormap
    from pathology_trainfree.data import read_manifest, load_image, load_mask, resize_mask
    metrics = read_csv(OUT / "per_image_metrics.csv")
    effects = read_csv(OUT / "paired_effects.csv")
    flips = read_csv(OUT / "decision_flip_analysis.csv")
    times = read_csv(OUT / "runtime_memory.csv")
    summary = read_csv(OUT / "summary.csv")
    figdir = OUT / "figures"; figdir.mkdir(exist_ok=True)
    colors = {"S0": "#555555", "S1": "#2a9d8f", "S2": "#457b9d", "S3": "#e76f51"}

    # Figure 1: macro performance.
    fields = [("dice", "Dice"), ("object_f1", "Object-F1"),
              ("boundary_f1", "Boundary-F1"), ("small_gland_recall", "Small gland recall")]
    fig, axs = plt.subplots(1, 4, figsize=(15, 4))
    for ax, (field, title) in zip(axs, fields):
        for xi, ds in enumerate(("GlaS", "CRAG")):
            vals = [fnum(next(r[field] for r in summary if r["dataset"] == ds and r["method"] == m)) for m in ("S0", "S1", "S2", "S3")]
            ax.plot(range(4), vals, marker="o", label=ds if field == "dice" else None)
        ax.set_xticks(range(4), ["S0\n2 phase", "S1\n4 phase", "S2\n16 ens", "S3\n16 stitch"])
        ax.set_title(title); ax.grid(alpha=.25)
    axs[0].legend(frameon=False); fig.tight_layout(); fig.savefig(figdir / "figure_1_overall_performance.png", dpi=180); plt.close(fig)

    # Figure 2: number of true phase observations versus performance.
    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    for ds in ("GlaS", "CRAG"):
        for method, n in (("S0", 2), ("S1", 4), ("S2", 16), ("S3", 16)):
            r = next(x for x in summary if x["dataset"] == ds and x["method"] == method)
            axs[0].scatter(n, fnum(r["dice"]), color=colors[method], marker="o" if ds == "GlaS" else "s", label=f"{ds} {method}")
            axs[1].scatter(n, fnum(r["object_f1"]), color=colors[method], marker="o" if ds == "GlaS" else "s")
    axs[0].set_ylabel("Dice"); axs[1].set_ylabel("Object-F1")
    for ax in axs: ax.set_xlabel("True UNI phases"); ax.set_xticks([2, 4, 16]); ax.grid(alpha=.25)
    axs[0].legend(fontsize=7, ncol=2); fig.tight_layout(); fig.savefig(figdir / "figure_2_sampling_vs_performance.png", dpi=180); plt.close(fig)

    # Figure 4: image-macro net corrected pixels by GT boundary distance.
    strata = ["boundary_0_2", "boundary_2_5", "interior_gt_5"]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for ax, ds in zip(axs, ("GlaS", "CRAG")):
        for comp, col in (("S1-S0", colors["S1"]), ("S2-S0", colors["S2"]), ("S3-S0", colors["S3"]), ("S3-S2", "#9467bd")):
            rr = [x for x in flips if x["dataset"] == ds and x["comparison"] == comp]
            ys = [np.mean([fnum(r[s + "_net"]) for r in rr]) for s in strata]
            ax.plot(range(3), ys, marker="o", label=comp, color=col)
        ax.axhline(0, color="black", lw=.7); ax.set_title(ds); ax.set_xticks(range(3), ["0–2 px", "2–5 px", ">5 px"]); ax.set_ylabel("Mean net corrected pixels / image"); ax.grid(alpha=.25)
    axs[1].legend(frameon=False); fig.tight_layout(); fig.savefig(figdir / "figure_4_boundary_error_localization.png", dpi=180); plt.close(fig)

    # Figure 5: observed full extraction cost and metrics.
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    for ds in ("GlaS", "CRAG"):
        rr = [x for x in times if x["dataset"] == ds]
        measured = [x for x in rr if str(x.get("feature_extraction_timing_complete", "true")).lower() == "true"]
        sec = np.mean([fnum(x["feature_extraction_seconds"]) for x in measured])
        actual_calls = np.mean([fnum(x["actual_uni_forward_batches"]) for x in rr])
        online_calls = np.mean([fnum(x["if_all_16_phases_fresh_uni_forward_batches"]) for x in rr])
        dice = fnum(next(x["dice"] for x in summary if x["dataset"] == ds and x["method"] == "S3"))
        obj = fnum(next(x["object_f1"] for x in summary if x["dataset"] == ds and x["method"] == "S3"))
        x = 0 if ds == "GlaS" else 1
        axs[0].bar(x-.17, actual_calls, width=.34, label="new phases actually run" if x == 0 else None)
        axs[0].bar(x+.17, online_calls, width=.34, label="all 16 online" if x == 0 else None)
        axs[1].bar(x, sec, color="#457b9d", label="measured 14 new phases" if x == 0 else None)
        axs[2].scatter(sec, dice, label=f"{ds} Dice")
        axs[2].scatter(sec, obj, marker="s", label=f"{ds} Object-F1")
    axs[0].set_xticks([0,1],["GLaS","CRAG"]); axs[0].set_ylabel("UNI model.forward batches / image"); axs[0].legend(fontsize=7); axs[0].grid(axis="y",alpha=.25)
    axs[1].set_xticks([0,1],["GLaS","CRAG"]); axs[1].set_ylabel("Measured UNI feature time / image (s)"); axs[1].legend(fontsize=7); axs[1].grid(axis="y",alpha=.25)
    axs[2].set(xlabel="Measured UNI feature time / image (s)", ylabel="Macro score"); axs[2].grid(alpha=.25); axs[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(figdir / "figure_5_performance_cost.png", dpi=180); plt.close(fig)

    # Figures 3 and 6: strongest S3 gain and loss per dataset, both shown.
    case_rows = read_csv(OUT / "failure_cases.csv")
    cmap = ListedColormap([(0,0,0,0), "#4daf4a", "#e41a1c", "#ffcc00"])
    for ds in ("GlaS", "CRAG"):
        md = ROOT / f"datasets/{ds}/manifests/paper_seed_2026/test.csv"
        entries = read_manifest(md, require_mask=True); by_id = {e["id"]: e for e in entries}
        (OUT / "cases" / ds).mkdir(parents=True, exist_ok=True)
        rows = [r for r in case_rows if r["dataset"] == ds]
        best = max(rows, key=lambda r: fnum(r["s3_minus_s0_dice"]))
        worst = min(rows, key=lambda r: fnum(r["s3_minus_s0_dice"]))
        for role, chosen in (("improvement", best), ("degradation", worst)):
            iid = chosen["image_id"]; entry = by_id[iid]
            rgb = np.asarray(load_image(entry["image"]))
            import cv2
            rgb = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_AREA)
            gt = resize_mask(load_mask(entry["mask"], {"mask_encoding": "indexed", "class_mapping": {0: 0, 255: 1}}), (256, 256)) > 0
            preds = {m: np.load(OUT / "predictions" / ds / f"{iid}_{m}.npz")["mask"].astype(bool) for m in ("S0", "S1", "S2", "S3")}
            margins = {m: np.load(OUT / "predictions" / ds / f"{iid}_{m}.npz")["margin"].astype(np.float32) for m in ("S0", "S1", "S2", "S3")}
            # Overlay: corrected green, damaged red, remaining GT boundary yellow.
            s0err, s3err = preds["S0"] != gt, preds["S3"] != gt
            overlay = np.zeros((256, 256), np.uint8)
            overlay[s0err & ~s3err] = 1; overlay[~s0err & s3err] = 2
            gy = gt & ~ndimage.binary_erosion(gt)
            overlay[gy] = 3
            panels = [(rgb, "RGB"), (gt, "GT"), (preds["S0"], "S0"), (preds["S1"], "S1"),
                      (preds["S2"], "S2"), (preds["S3"], "S3"), (margins["S3"]-margins["S2"], "S3 − S2 margin"), ((rgb, overlay), "Corrected / damaged")]
            fig, axs = plt.subplots(2, 4, figsize=(13, 7))
            for ax, (im, title) in zip(axs.flat, panels):
                if title == "Corrected / damaged":
                    ax.imshow(im[0]); ax.imshow(im[1], cmap=cmap, vmin=0, vmax=3, alpha=.8)
                elif title == "S3 − S2 margin": ax.imshow(im, cmap="coolwarm", vmin=-max(1e-6,np.max(np.abs(im))), vmax=max(1e-6,np.max(np.abs(im))))
                else: ax.imshow(im, cmap="gray" if np.asarray(im).ndim == 2 else None)
                ax.set_title(title); ax.axis("off")
            fig.suptitle(f"{ds} {role}: {iid}, S3−S0 Dice={fnum(chosen['s3_minus_s0_dice']):+.4f}")
            fig.tight_layout(); fig.savefig(OUT / "cases" / ds / f"{iid}_{role}.png", dpi=160)
            if role == "improvement": fig.savefig(figdir / f"figure_3_{ds}_ensemble_vs_stitch.png", dpi=160)
            else: fig.savefig(figdir / f"figure_6_{ds}_representative_cases.png", dpi=160)
            plt.close(fig)
    print("Wrote figures and case panels")


if __name__ == "__main__": main()
