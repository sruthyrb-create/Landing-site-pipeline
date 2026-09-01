"""
Synthetic terrain for the Malapert Massif case study ONLY.

This module exists because the real high-resolution raster DEM for the
lunar south pole (NASA GSFC PGDA's 5 m/pixel LOLA topography for candidate
Artemis III sites) was not reachable from this sandboxed environment's
network (see README "Known limitations"). To still be able to demonstrate
the full pipeline -- crater detection, slope/roughness, illumination,
composite scoring -- on a real, named, aerospace-relevant polar site, this
module builds a labeled *synthetic* elevation surface that is informed by
two real, citable facts rather than invented from nothing:

  1. The real positions and diameters of every Head et al. (2010) catalog
     crater near the site (these are genuine craters at their genuine
     locations), rendered as bowls using a standard published lunar
     depth/diameter relation (Pike, 1974, 1977):
         simple craters   (D <  15 km): depth = 0.20 * D
         complex craters  (D >= 15 km): depth = 1.044 * D**0.301
     (depth and D both in km).

  2. Malapert Mountain's published description as a ridge roughly 5 km
     above its surroundings on the southwest rim of Malapert crater
     (per USGS/lunar-reference descriptions cited in the README),
     rendered as a smooth raised massif at its approximate location.

  A gently rolling highland background (smoothed random noise, amplitude
  chosen to match typical lunar highland roughness of a few hundred
  meters) fills in the rest of the surface.

Geometry: built directly as a square box in real kilometers (a local flat
approximation), NOT as a square box in lat/lon degrees -- this close to the
pole, a fixed-degree box in longitude covers a tiny real distance while the
same degree range in latitude covers a huge one (see regions.py). Lat/lon
are only computed afterwards, per pixel, for labeling/plotting.

This is explicitly a teaching aid, not a measurement. Every figure derived
from it in outputs/plots is labeled "(synthetic terrain)".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from .data import DemCrop, MOON_RADIUS_M


def latlon_to_local_km(lat_center: float, lon_center: float, lat, lon):
    """Flat-local-approximation lat/lon -> (x_km east, y_km north) offset from center."""
    dy_km = (np.asarray(lat, dtype=float) - lat_center) * (np.pi / 180.0) * MOON_RADIUS_M / 1000.0
    dx_km = (
        (np.asarray(lon, dtype=float) - lon_center)
        * (np.pi / 180.0) * MOON_RADIUS_M / 1000.0 * np.cos(np.radians(lat_center))
    )
    return dx_km, dy_km


def local_km_to_latlon(lat_center: float, lon_center: float, dx_km, dy_km):
    """Inverse of latlon_to_local_km."""
    lat = lat_center + np.asarray(dy_km, dtype=float) / (np.pi * MOON_RADIUS_M / 1000.0 / 180.0)
    lon = lon_center + np.asarray(dx_km, dtype=float) / (
        np.pi * MOON_RADIUS_M / 1000.0 / 180.0 * np.cos(np.radians(lat_center))
    )
    return lat, lon


def _crater_depth_km(diam_km: np.ndarray) -> np.ndarray:
    diam_km = np.asarray(diam_km, dtype=float)
    simple = 0.20 * diam_km
    complex_ = 1.044 * np.power(diam_km, 0.301)
    return np.where(diam_km < 15.0, simple, complex_)


def build_synthetic_dem(
    lat_center: float,
    lon_center: float,
    half_width_km: float,
    craters_window: pd.DataFrame,
    grid_size: int = 400,
    highland_roughness_m: float = 250.0,
    massif_offset_km: tuple[float, float] = (-6.0, 4.0),
    massif_height_km: float = 5.0,
    massif_radius_km: float = 12.0,
    seed: int = 7,
) -> DemCrop:
    """
    Build a labeled synthetic DemCrop for a polar site lacking raster data.

    craters_window craters are placed by their true lat/lon, converted to
    local (x_km, y_km) offsets from (lat_center, lon_center) using a flat
    local approximation -- craters far outside the box are simply clipped
    away by the bowl's own footprint not overlapping the grid.
    """
    rng = np.random.default_rng(seed)

    mpp = (2 * half_width_km * 1000.0) / grid_size  # meters/pixel, same in both directions

    base = rng.normal(0, 1, size=(grid_size, grid_size))
    base = gaussian_filter(base, sigma=grid_size / 40.0)
    base = (base - base.mean()) / (base.std() + 1e-9) * highland_roughness_m

    ys, xs = np.mgrid[0:grid_size, 0:grid_size]
    x_km = (xs - grid_size / 2) * mpp / 1000.0
    y_km = (grid_size / 2 - ys) * mpp / 1000.0  # row 0 = "north" (+y) within this local box

    elevation_m = base.copy()

    for _, row in craters_window.iterrows():
        cx_km, cy_km = latlon_to_local_km(lat_center, lon_center, row["lat"], row["lon"])
        radius_km = row["diam_km"] / 2.0
        depth_m = _crater_depth_km(row["diam_km"]) * 1000.0
        r = np.hypot(x_km - cx_km, y_km - cy_km)
        profile = np.where(
            r <= radius_km,
            -depth_m * 0.5 * (1 + np.cos(np.pi * r / radius_km)),
            0.0,
        )
        elevation_m += profile

    mx_km, my_km = massif_offset_km
    r = np.hypot(x_km - mx_km, y_km - my_km)
    dome = massif_height_km * 1000.0 * np.exp(-0.5 * (r / massif_radius_km) ** 2)
    elevation_m += dome

    # Corner lat/lon bounds, kept only as metadata (NOT used for plot axes --
    # see is_local_km on DemCrop and viz.py).
    lat_south, lon_west = local_km_to_latlon(lat_center, lon_center, -half_width_km, -half_width_km)
    lat_north, lon_east = local_km_to_latlon(lat_center, lon_center, half_width_km, half_width_km)

    return DemCrop(
        elevation_m=elevation_m,
        lat_center=lat_center,
        lon_center=lon_center,
        half_width_deg=float("nan"),
        lat_bounds=(lat_south, lat_north),
        lon_bounds=(lon_west, lon_east),
        mpp_ns=mpp,
        mpp_ew=mpp,
        is_local_km=True,
    )
