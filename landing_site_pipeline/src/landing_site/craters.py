"""
Classical computer-vision crater detection on real DEM data, plus validation
against the Head et al. (2010) published crater catalog.

Method
------
Craters are topographic bowls: locally low elevation surrounded by a rim.
That makes them detectable as blobs in scale-space on the *negated*
elevation surface (so crater floors become local maxima), using the
Laplacian-of-Gaussian blob detector (`skimage.feature.blob_log`). This is a
classical-CV baseline deliberately kept simple and explainable -- the
YOLOv8 fine-tuning path in this project is an optional upgrade over this
baseline, not a replacement for understanding it.

Validation
----------
The Head catalog only contains craters >= 20 km diameter, so we only score
precision/recall against catalog entries in that size range. Detections
below 20 km are kept and reported separately as "uncatalogued candidates"
-- this mirrors how real crater surveys are described in the literature
(smaller craters are systematically undercounted in older/coarser global
catalogs), rather than treating "not in the catalog" as "wrong."
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from skimage.feature import blob_log

from .data import DemCrop


@dataclass
class CraterDetections:
    x_px: np.ndarray
    y_px: np.ndarray
    radius_km: np.ndarray
    lat: np.ndarray
    lon: np.ndarray


@dataclass
class ValidationResult:
    n_catalog_in_window: int
    n_matched: int
    n_unmatched_catalog: int
    n_uncatalogued_candidates: int
    precision_ge20km: float
    recall_ge20km: float
    matches: pd.DataFrame = field(repr=False)


def detect_craters(
    dem_crop: DemCrop,
    min_radius_km: float = 5.0,
    max_radius_km: float = 60.0,
    n_radii: int = 16,
    threshold: float = 0.03,
) -> CraterDetections:
    """Detect crater-like circular depressions via LoG blob detection."""
    elevation = dem_crop.elevation_m
    mpp = (dem_crop.mpp_ns + dem_crop.mpp_ew) / 2.0

    # Normalize to [0, 1] and negate so crater floors (locally low) -> bright blobs.
    e = elevation.astype(np.float64)
    e_norm = (e - e.min()) / max(e.max() - e.min(), 1e-9)
    inverted = 1.0 - e_norm

    min_sigma = (min_radius_km * 1000.0 / mpp) / np.sqrt(2)
    max_sigma = (max_radius_km * 1000.0 / mpp) / np.sqrt(2)
    min_sigma = max(min_sigma, 1.0)

    blobs = blob_log(
        inverted,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=n_radii,
        threshold=threshold,
    )
    if len(blobs) == 0:
        empty = np.array([])
        return CraterDetections(empty, empty, empty, empty, empty)

    y_px, x_px, sigma = blobs[:, 0], blobs[:, 1], blobs[:, 2]
    radius_px = sigma * np.sqrt(2)
    radius_km = radius_px * mpp / 1000.0

    south, north = dem_crop.lat_bounds
    west, east = dem_crop.lon_bounds
    h, w = elevation.shape
    lat = north - (y_px / h) * (north - south)
    lon = west + (x_px / w) * (east - west)

    return CraterDetections(x_px=x_px, y_px=y_px, radius_km=radius_km, lat=lat, lon=lon)


def validate_against_catalog(
    detections: CraterDetections,
    catalog_window: pd.DataFrame,
    match_radius_km: float = 15.0,
    catalog_min_diam_km: float = 20.0,
) -> ValidationResult:
    """
    Greedy nearest-neighbor matching between detections and catalog craters
    (restricted to the catalog's actual size floor) using simple great-circle-ish
    small-angle distance in km. Good enough at regional scale; not meant for
    global-scale matching.
    """
    catalog = catalog_window[catalog_window["diam_km"] >= catalog_min_diam_km].reset_index(drop=True)

    matches = []
    matched_catalog_idx = set()
    matched_detection_idx = set()

    if len(detections.lat) and len(catalog):
        lat_c = np.radians(catalog["lat"].to_numpy())
        for i in range(len(detections.lat)):
            dlat = catalog["lat"].to_numpy() - detections.lat[i]
            dlon = (catalog["lon"].to_numpy() - detections.lon[i]) * np.cos(lat_c)
            dist_km = np.hypot(dlat, dlon) * (np.pi / 180.0) * 1737.4
            j = int(np.argmin(dist_km)) if len(dist_km) else None
            if j is not None and dist_km[j] <= match_radius_km and j not in matched_catalog_idx:
                matches.append(
                    {
                        "detection_idx": i,
                        "catalog_idx": j,
                        "distance_km": dist_km[j],
                        "detected_radius_km": detections.radius_km[i],
                        "catalog_diam_km": catalog.loc[j, "diam_km"],
                    }
                )
                matched_catalog_idx.add(j)
                matched_detection_idx.add(i)

    n_catalog = len(catalog)
    n_matched = len(matches)
    n_unmatched_catalog = n_catalog - n_matched
    n_detections = len(detections.lat)
    n_uncatalogued = n_detections - n_matched

    precision = n_matched / n_detections if n_detections else float("nan")
    recall = n_matched / n_catalog if n_catalog else float("nan")

    return ValidationResult(
        n_catalog_in_window=n_catalog,
        n_matched=n_matched,
        n_unmatched_catalog=n_unmatched_catalog,
        n_uncatalogued_candidates=n_uncatalogued,
        precision_ge20km=precision,
        recall_ge20km=recall,
        matches=pd.DataFrame(matches),
    )


def crater_density_map(
    dem_crop: DemCrop,
    detections: CraterDetections,
    grid_cells: int = 20,
) -> np.ndarray:
    """Coarse-grid crater count density (craters per grid cell) for scoring."""
    h, w = dem_crop.shape
    density = np.zeros((grid_cells, grid_cells))
    if len(detections.x_px) == 0:
        return density
    xi = np.clip((detections.x_px / w * grid_cells).astype(int), 0, grid_cells - 1)
    yi = np.clip((detections.y_px / h * grid_cells).astype(int), 0, grid_cells - 1)
    # Weight bigger craters more heavily -- they matter more for landing hazard.
    weights = np.clip(detections.radius_km / 5.0, 1.0, None)
    for x, y, wgt in zip(xi, yi, weights):
        density[y, x] += wgt
    return density
