"""
Composite landing-site suitability scoring.

Combines three normalized layers into one 0-100 suitability score per grid
cell:

    suitability = 100 * (w_crater * (1 - crater_hazard)
                        + w_slope  * (1 - slope_hazard)
                        + w_illum  * illumination_benefit)

where each hazard/benefit term is min-max normalized to [0, 1] within the
analyzed region before weighting, and the weights sum to 1. Weights are
deliberately exposed as parameters rather than hard-coded, because the
"right" weighting is a genuine engineering/mission-design judgment call
(a crewed lander cares differently about slope than an uncrewed rover
does) -- the README discusses this explicitly rather than presenting one
fixed formula as ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import zoom


def _normalize01(arr: np.ndarray) -> np.ndarray:
    lo, hi = np.nanmin(arr), np.nanmax(arr)
    if hi - lo < 1e-9:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


@dataclass
class SuitabilityWeights:
    crater: float = 0.35
    slope: float = 0.35
    illumination: float = 0.30

    def normalized(self) -> "SuitabilityWeights":
        total = self.crater + self.slope + self.illumination
        return SuitabilityWeights(self.crater / total, self.slope / total, self.illumination / total)


def compute_suitability(
    crater_density: np.ndarray,
    slope_deg: np.ndarray,
    illumination_score: np.ndarray,
    weights: SuitabilityWeights = SuitabilityWeights(),
) -> np.ndarray:
    """
    All three inputs may be on different grid resolutions (crater_density is
    typically a coarse grid; slope_deg/illumination_score are per-pixel).
    Everything is resampled to the shape of `slope_deg` before combining.
    """
    w = weights.normalized()
    target_shape = slope_deg.shape

    def resample(arr):
        if arr.shape == target_shape:
            return arr
        zy = target_shape[0] / arr.shape[0]
        zx = target_shape[1] / arr.shape[1]
        return zoom(arr, (zy, zx), order=1)

    crater_norm = _normalize01(resample(crater_density))
    slope_norm = _normalize01(resample(slope_deg))
    illum_norm = _normalize01(resample(illumination_score))

    suitability = 100.0 * (
        w.crater * (1 - crater_norm) + w.slope * (1 - slope_norm) + w.illumination * illum_norm
    )
    return suitability


def best_site(suitability: np.ndarray, dem_crop) -> dict:
    """Return the lat/lon and score of the single best-scoring pixel."""
    idx = np.unravel_index(np.nanargmax(suitability), suitability.shape)
    h, w = suitability.shape
    south, north = dem_crop.lat_bounds
    west, east = dem_crop.lon_bounds
    lat = north - (idx[0] / h) * (north - south)
    lon = west + (idx[1] / w) * (east - west)
    return {"lat": lat, "lon": lon, "score": float(suitability[idx])}
