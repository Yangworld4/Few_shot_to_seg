"""Optional SAM refinement and its paired evaluation outputs."""

from ..core.sam_refinement import cleanup_mask, refine_instances
from ..core.selfref_geometry import reconstruct_native
from ..models.sam import load_sam_predictor
from ..selfref_config import ROOT
from ..selfref_data import read_original_image, shape
from ..utils.selfref_metrics import compare_sam


class SamRefinement:
    fields = ("sam_attempted", "sam_accepted")

    def __init__(self, cfg):
        self.settings = cfg["sam"]
        self.baseline = self.settings["base_branch"]
        self.branches = (f"{self.baseline}_cleanup", f"{self.baseline}_sam")
        self.reconstruction = cfg["reconstruction"]
        self.threshold = cfg["threshold"]
        self.predictor = load_sam_predictor(ROOT, self.settings["checkpoint"],
                                            self.settings["model_type"], cfg["device"])

    def start_query(self, row):
        self.predictor.set_image(read_original_image(row))

    def prepare(self, margins, positions, row):
        cleanup, refined = self.branches
        native = reconstruct_native(margins[self.baseline], positions, shape(row), self.reconstruction)
        mask = native > self.threshold
        updated, stats = refine_instances(self.predictor, native, mask, self.threshold, self.settings)
        hole_area, min_area = self.settings["max_hole_area"], self.settings["min_area"]
        predictions = {self.baseline: (native, mask), cleanup: (None, cleanup_mask(mask, hole_area, min_area)),
                       refined: (None, cleanup_mask(updated, hole_area, min_area))}
        return predictions, {refined: stats}

    def comparison(self, rows):
        return compare_sam(rows, self.baseline, *self.branches)
