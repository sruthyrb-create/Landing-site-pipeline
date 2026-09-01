"""
Extension analysis: (1) an alternative multi-criteria combination method
(TOPSIS) to compare against the ad hoc weighted sum in scoring.py, and
(2) a one-at-a-time (OAT) sensitivity analysis of the weighted sum's
weights.

Motivation
----------
The composite score in scoring.py combines three layers with a fixed
weighted sum (crater=0.35, slope=0.35, illumination=0.30). This is a
reasonable, explicit default -- but it is a design choice, not a derived
fact, and the report says so. Two standard techniques make that statement
checkable rather than just asserted:

1. TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)
   is the same class of formal multi-criteria decision-making method used
   by Pena-Asensio, Neira-Acosta & Sanchez-Lozano (2024) to rank real
   Artemis III candidate sites. Applying it here, on the same three
   layers already computed, and comparing its chosen best site against
   the weighted sum's, tests whether the two reasonable-but-different
   combination methods actually agree.

2. One-at-a-time (OAT) sensitivity analysis is a standard, named technique
   for testing how much a model's output depends on an uncertain
   parameter: vary one weight across its full range (redistributing the
   remaining budget proportionally between the other two, so weights
   always sum to 1), holding everything else fixed, and record how much
   the best site's score and location move. A large shift means the
   "best" site is fragile and depends heavily on a subjective weighting
   choice; a small shift means the result is robust to that choice.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.ndimage import zoom

from .data import DemCrop
from .scoring import SuitabilityWeights, _normalize01, best_site
from .synthetic import latlon_to_local_km

MOON_RADIUS_KM = 1737.4


def _site_distance_km(dem_crop: DemCrop, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Real-distance-in-km between two sites within the same crop, using
    whichever coordinate system that crop actually uses (see is_local_km).
    Uses the same small-angle approximation as craters.py's catalog
    matching for real (lat/lon) crops, for consistency across the project.
    """
    if dem_crop.is_local_km:
        x1, y1 = latlon_to_local_km(dem_crop.lat_center, dem_crop.lon_center, lat1, lon1)
        x2, y2 = latlon_to_local_km(dem_crop.lat_center, dem_crop.lon_center, lat2, lon2)
        return float(np.hypot(x1 - x2, y1 - y2))
    dlat = lat2 - lat1
    dlon = (lon2 - lon1) * np.cos(np.radians((lat1 + lat2) / 2.0))
    return float(np.hypot(dlat, dlon) * (np.pi / 180.0) * MOON_RADIUS_KM)


def _resample_all(crater_density, slope_deg, illumination_score):
    """Same resampling behavior as scoring.compute_suitability, exposed
    separately so both TOPSIS and the sensitivity sweep can reuse it
    without recomputing it many times."""
    target_shape = slope_deg.shape

    def resample(arr):
        if arr.shape == target_shape:
            return arr
        zy = target_shape[0] / arr.shape[0]
        zx = target_shape[1] / arr.shape[1]
        return zoom(arr, (zy, zx), order=1)

    return resample(crater_density), slope_deg, resample(illumination_score)


def compute_topsis(
    crater_density: np.ndarray,
    slope_deg: np.ndarray,
    illumination_score: np.ndarray,
    weights: SuitabilityWeights = SuitabilityWeights(),
) -> np.ndarray:
    """
    Combine the same three layers scoring.compute_suitability uses, but
    with TOPSIS instead of a weighted sum -- the multi-criteria decision-
    making method used by Pena-Asensio et al. (2024) for the same class of
    problem. Returns a per-pixel "closeness to the ideal solution" score
    in [0, 1] (higher = better), directly comparable in shape to
    scoring.compute_suitability's output (best_site() works on either).

    TOPSIS steps, applied treating each pixel as one "alternative":
      1. Vector-normalize each criterion layer (divide by its Euclidean
         norm across all pixels), the standard TOPSIS normalization
         (different from scoring.py's min-max normalization).
      2. Weight each normalized criterion by its importance weight.
      3. For each criterion, find the "ideal best" and "ideal worst"
         value across all pixels -- for a hazard (crater, slope) the
         ideal best is the MINIMUM; for a benefit (illumination) the
         ideal best is the MAXIMUM.
      4. For each pixel, compute its Euclidean distance to the ideal-best
         point and to the ideal-worst point across all three criteria.
      5. Closeness = distance-to-worst / (distance-to-best + distance-to-worst).
         A pixel simultaneously close to the ideal best and far from the
         ideal worst scores near 1.
    """
    w = weights.normalized()
    crater, slope, illum = _resample_all(crater_density, slope_deg, illumination_score)

    criteria = {
        "crater": (crater, "cost", w.crater),
        "slope": (slope, "cost", w.slope),
        "illumination": (illum, "benefit", w.illumination),
    }

    weighted = {}
    for name, (arr, _, weight) in criteria.items():
        norm_factor = np.sqrt(np.nansum(arr.astype(np.float64) ** 2))
        norm_factor = norm_factor if norm_factor > 1e-9 else 1.0
        weighted[name] = weight * (arr / norm_factor)

    ideal_best = {}
    ideal_worst = {}
    for name, (_, kind, _) in criteria.items():
        arr = weighted[name]
        if kind == "cost":
            ideal_best[name] = np.nanmin(arr)
            ideal_worst[name] = np.nanmax(arr)
        else:
            ideal_best[name] = np.nanmax(arr)
            ideal_worst[name] = np.nanmin(arr)

    dist_best_sq = sum((weighted[n] - ideal_best[n]) ** 2 for n in criteria)
    dist_worst_sq = sum((weighted[n] - ideal_worst[n]) ** 2 for n in criteria)
    dist_best = np.sqrt(dist_best_sq)
    dist_worst = np.sqrt(dist_worst_sq)

    denom = dist_best + dist_worst
    denom = np.where(denom < 1e-9, 1.0, denom)
    closeness = dist_worst / denom
    return closeness * 100.0  # rescale to a 0-100 range like scoring.compute_suitability


@dataclass
class SensitivityResult:
    varied_criterion: str
    weight_values: np.ndarray       # the swept weight (0..1)
    best_scores: np.ndarray         # best-site score at each swept weight
    shift_km: np.ndarray            # distance of the new best site from the baseline best site


def one_at_a_time_sensitivity(
    crater_density: np.ndarray,
    slope_deg: np.ndarray,
    illumination_score: np.ndarray,
    dem_crop: DemCrop,
    base_weights: SuitabilityWeights = SuitabilityWeights(),
    n_steps: int = 11,
) -> tuple[dict, list[SensitivityResult]]:
    """
    One-at-a-time sensitivity analysis of the weighted-sum composite score
    (scoring.compute_suitability). For each of the three weights in turn,
    sweep it from 0 to 1 in n_steps, redistributing the remaining budget
    between the other two weights in their default proportion (so all
    three still sum to 1 at every step), and record how far the new
    best-scoring site is (in real km) from the baseline (default-weight)
    best site, and what its score is.

    Returns (baseline_best, [SensitivityResult, ...]) -- one
    SensitivityResult per criterion ("crater", "slope", "illumination").
    """
    from .scoring import compute_suitability  # local import avoids a cycle at module load

    base = base_weights.normalized()
    baseline_suit = compute_suitability(crater_density, slope_deg, illumination_score, base)
    baseline_best = best_site(baseline_suit, dem_crop)

    names = ["crater", "slope", "illumination"]
    results = []

    for name in names:
        others = [n for n in names if n != name]
        default_vals = {n: getattr(base, n) for n in names}
        other_total = default_vals[others[0]] + default_vals[others[1]]
        ratio0 = default_vals[others[0]] / other_total if other_total > 1e-9 else 0.5
        ratio1 = 1.0 - ratio0

        weight_values = np.linspace(0.0, 1.0, n_steps)
        best_scores = np.zeros(n_steps)
        shift_km = np.zeros(n_steps)

        for i, v in enumerate(weight_values):
            remaining = 1.0 - v
            kwargs = {name: v, others[0]: remaining * ratio0, others[1]: remaining * ratio1}
            w = SuitabilityWeights(**kwargs)
            suit = compute_suitability(crater_density, slope_deg, illumination_score, w)
            best = best_site(suit, dem_crop)
            best_scores[i] = best["score"]
            shift_km[i] = _site_distance_km(
                dem_crop, baseline_best["lat"], baseline_best["lon"], best["lat"], best["lon"]
            )

        results.append(
            SensitivityResult(
                varied_criterion=name,
                weight_values=weight_values,
                best_scores=best_scores,
                shift_km=shift_km,
            )
        )

    return baseline_best, results
