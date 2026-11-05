"""Small preregistered geometry assertions (no image labels or UNI forward)."""
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pathology_trainfree.shift_stitch import phase_grid, windows_xy, token_centers, reflect_coordinate, dense_stitch


def check():
    p4 = phase_grid((0, 8)); p16 = phase_grid((0, 4, 8, 12))
    assert len(set(p4)) == 4 and len(set(p16)) == 16
    sample = np.arange(32 * 40, dtype=np.uint16).reshape(32, 40)
    positions = {}
    for phase in p4:
        views = list(windows_xy(sample, 32, 16, phase, "reflect"))
        assert views[0][:2] == (-phase[0], -phase[1])
        assert all(c.shape == (32, 32) for _, _, c in views)
        assert np.array_equal(views[0][2][0, 0], sample[min(phase[0], 31), min(phase[1], 39)])
        positions[phase] = np.asarray([(y, x) for y, x, _ in views])
    # Independent X/Y offsets create all four relative phase origins.
    centers = {p: token_centers(np.array([[ -p[0], -p[1] ]]), 1, 16) for p in p16}
    x0 = sorted({float(centers[p][0][0, 0]) for p in p16})
    assert np.allclose(np.diff(x0), 4)
    assert np.allclose(reflect_coordinate([-1, 0, 1, 39, 40], 40), [1, 0, 1, 39, 38])

    # Analytic reflected linear field on a complete 4px lattice.
    phases = phase_grid((0, 4, 8, 12))
    image = np.zeros((64, 72, 3), np.uint8)
    grids, pos = {}, {}
    for p in phases:
        views = list(windows_xy(image, 32, 16, p, "reflect"))
        pos[p] = np.asarray([(y, x) for y, x, _ in views])
        xx, yy = token_centers(pos[p], grid=2, patch=16)
        # Each window has the same local grid; its actual origin is included.
        values = []
        for wi, (y, x) in enumerate(pos[p]):
            xcoords, ycoords = token_centers(np.asarray([[y, x]]), grid=2, patch=16)
            mx, my = reflect_coordinate(xcoords[0], 40), reflect_coordinate(ycoords[0], 32)
            mxg, myg = np.meshgrid(mx, my)
            values.append((mxg + myg).astype(np.float16))
        grids[p] = np.stack(values)
    _, dense, diag, (_, _, _, counts) = dense_stitch(grids, pos, phases, (64, 72), grid=2, patch=16, output_size=(64, 72))
    assert dense.shape == (64, 72) and np.isfinite(dense).all()
    assert diag["dense_coverage"] == 1.0 and diag["overlap_observations"] > 0
    assert np.count_nonzero(counts) > 0 and diag["x_pitch_median"] == 4.0 and diag["y_pitch_median"] == 4.0
    return {"status": "PASS", "phase_count_4": len(p4), "phase_count_16": len(p16),
            "x_pitch_median": diag["x_pitch_median"], "y_pitch_median": diag["y_pitch_median"],
            "unique_reflected_points": diag["unique_reflected_positions"],
            "overlap_observations": diag["overlap_observations"],
            "dense_grid_coverage": diag["dense_coverage"],
            "missing_native_pixels": diag["missing_native_pixels_before_extension"]}


if __name__ == "__main__": print(check())
