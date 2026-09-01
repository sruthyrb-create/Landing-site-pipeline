"""
Two illumination models, used for two different scientific situations.

1. `mid_latitude_incidence()` -- for REGION_TYCHO (43S), where the Moon's
   tiny axial tilt (1.54 deg) means the sun's elevation stays moderate
   (roughly 35-55 deg depending on local time and lunar season) and never
   grazes the horizon. Here the dominant illumination effect is simply the
   cosine law: a slope facing the sun gets more direct light per unit area
   than a slope facing away from it. This is computed directly and exactly
   from the real DEM's slope and aspect -- no approximation beyond the
   standard Lambertian cosine-incidence model.

2. `polar_illumination_proxy()` -- for REGION_MALAPERT (~85S), where the
   physics are qualitatively different: the sun's elevation angle is always
   within about 1.5 deg of the local horizon, so whether a point is lit
   depends almost entirely on whether *nearby terrain blocks the sun*, not
   on slope-facing geometry. This is the real mechanism behind the Moon's
   permanently-shadowed regions (crater floors near the pole that never see
   direct sunlight) and its rare near-continuously-illuminated high points
   ("peaks of eternal light") -- elevated massifs like Malapert stand above
   the horizon-blocking terrain around them and so catch far more sunlight
   over a lunar day than the crater floors nearby.

   Doing this exactly requires ray-tracing the true local horizon in every
   solar-azimuth direction across a lunar day against a real high-resolution
   DEM (this is what NASA GSFC's PGDA illumination products, and the 5
   m/pixel south-pole LOLA DEMs referenced in the README, are built to
   support). Because this sandboxed environment could not reach the PDS/PGDA
   data servers to download that raster (see README "Known limitations"),
   `polar_illumination_proxy()` instead estimates a *relative* illumination
   score from the point's height above the local median terrain height,
   which is the first-order physical driver of the real effect (a taller
   point has a physically higher local horizon angle it needs to clear).
   This is explicitly a proxy for teaching/demonstration, not a
   radiometrically validated illumination product -- the README explains
   exactly how to swap in the real DEM and the real ray-traced calculation
   once you have unrestricted network access.
"""

from __future__ import annotations

import numpy as np

from .data import DemCrop
from .terrain import compute_slope_deg


def _aspect_deg(elevation_m: np.ndarray, mpp_ns: float, mpp_ew: float) -> np.ndarray:
    """Downslope-facing direction in degrees, 0=north, 90=east (compass-style)."""
    dz_dy, dz_dx = np.gradient(elevation_m, mpp_ns, mpp_ew)
    aspect = np.degrees(np.arctan2(dz_dx, -dz_dy)) % 360.0
    return aspect


def mid_latitude_incidence(
    dem_crop: DemCrop,
    sun_azimuth_deg: float = 180.0,
    sun_elevation_deg: float = 45.0,
) -> np.ndarray:
    """
    Cosine-incidence illumination score in [0, 1] for a mid-latitude region,
    given a single representative sun position (azimuth measured clockwise
    from north, elevation above the horizon). Defaults approximate a
    plausible local-noon sun position at Tycho's latitude.
    """
    elevation = dem_crop.elevation_m
    slope_deg = compute_slope_deg(elevation, dem_crop.mpp_ns, dem_crop.mpp_ew)
    aspect_deg = _aspect_deg(elevation, dem_crop.mpp_ns, dem_crop.mpp_ew)

    slope = np.radians(slope_deg)
    aspect = np.radians(aspect_deg)
    sun_az = np.radians(sun_azimuth_deg)
    sun_el = np.radians(sun_elevation_deg)

    # Standard Lambertian cosine-of-incidence-angle model.
    cos_i = np.sin(sun_el) * np.cos(slope) + np.cos(sun_el) * np.sin(slope) * np.cos(sun_az - aspect)
    return np.clip(cos_i, 0.0, 1.0)


def polar_illumination_proxy(
    dem_crop: DemCrop,
    reference_percentile: float = 50.0,
    height_scale_km: float = 3.0,
) -> np.ndarray:
    """
    Relative illumination-likelihood proxy in [0, 1] for a near-polar region
    without a raster DEM available. Height above the local median terrain
    height is mapped through a saturating function -- taller points score
    higher, matching the real qualitative behavior of "peaks of eternal
    light," but the absolute values are NOT a physically calibrated
    illumination fraction. See module docstring.
    """
    elevation_km = dem_crop.elevation_m / 1000.0
    reference = np.percentile(elevation_km, reference_percentile)
    relative_height_km = elevation_km - reference
    return 1.0 / (1.0 + np.exp(-relative_height_km / (height_scale_km / 4.0)))
