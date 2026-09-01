"""
Slope and roughness derived directly from real elevation data.

Both quantities are computed from simple, transparent numpy operations on
the DEM crop so the method is easy to audit and explain in a report --
no black-box GIS calls.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import DemCrop


@dataclass
class TerrainMetrics:
    slope_deg: np.ndarray       # per-pixel slope magnitude, degrees
    roughness_m: np.ndarray     # local elevation std-dev within a small window, meters


def compute_slope_deg(elevation_m: np.ndarray, mpp_ns: float, mpp_ew: float) -> np.ndarray:
    """
    Per-pixel terrain slope in degrees, from the gradient of the elevation
    surface. dz/dy and dz/dx are estimated with a central-difference
    (np.gradient), each scaled by the real ground distance one pixel step
    represents in that direction.
    """
    dz_dy, dz_dx = np.gradient(elevation_m, mpp_ns, mpp_ew)
    slope_rad = np.arctan(np.hypot(dz_dx, dz_dy))
    return np.degrees(slope_rad)


def compute_roughness_m(elevation_m: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Local terrain roughness as the standard deviation of elevation within a
    (window x window) neighborhood -- a simple, standard terrain-ruggedness
    proxy used in rover/lander trafficability studies.
    """
    from scipy.ndimage import generic_filter

    half = window // 2
    padded = np.pad(elevation_m, half, mode="reflect")

    # A separable local-stddev via local mean of squares minus square of local mean
    # (much faster than generic_filter with a python callable).
    from scipy.ndimage import uniform_filter

    mean = uniform_filter(padded, size=window)
    mean_sq = uniform_filter(padded**2, size=window)
    var = np.clip(mean_sq - mean**2, 0, None)
    std = np.sqrt(var)
    return std[half:-half, half:-half] if half > 0 else std


def compute_terrain_metrics(dem_crop: DemCrop, roughness_window: int = 5) -> TerrainMetrics:
    slope = compute_slope_deg(dem_crop.elevation_m, dem_crop.mpp_ns, dem_crop.mpp_ew)
    roughness = compute_roughness_m(dem_crop.elevation_m, window=roughness_window)
    return TerrainMetrics(slope_deg=slope, roughness_m=roughness)
