"""
Loader for the real NASA GSFC PGDA 5 m/pixel south-pole LOLA DEM products
(pgda.gsfc.nasa.gov/products/78, pgda.gsfc.nasa.gov/data/LOLA_5mpp/).

These files are real GeoTIFFs in south polar stereographic coordinates
(meters, spherical Moon datum, MOON_ME/DE421 frame) -- NOT the equirectangular
lat/lon grid data.py's GlobalDem expects. This module reads them directly in
their native projection (rather than reprojecting to lat/lon, which would be
wasteful and introduce resampling error) and returns a DemCrop with
is_local_km=True, so every downstream module (craters.py, terrain.py,
illumination.py, scoring.py, sensitivity.py, viz.py) works completely
unmodified -- exactly the same drop-in path synthetic.py already established
for the Malapert region.
"""

from __future__ import annotations

import numpy as np
import rasterio
from pyproj import CRS, Transformer
from rasterio.enums import Resampling
from rasterio.windows import Window

from .data import DemCrop


def raster_footprint_latlon(tif_path: str) -> dict:
    """
    Report a PGDA site raster's OWN true center (lat, lon) and how much
    half-width (km) it actually covers, straight from its embedded
    projection/bounds -- no assumed site coordinates involved.

    Why this exists: the PGDA per-site 5 m/pixel tiles (Site23 etc.) are
    small, fixed footprints built around each candidate landing ellipse --
    NOT a generously-sized regional mosaic you can crop anywhere within.
    Site23_final_adj_5mpp_surf.tif, for example, is only ~21 x 21 km,
    centered at roughly 86.0 S, -0.2 E -- about 33 km from the (-84.9, 0.4)
    "Malapert Massif" coordinates regions.py uses for the *synthetic*
    stand-in terrain (which was built as a general-purpose illustrative
    site, not surveyed to this specific tile's footprint). Requesting a
    crop centered on the synthetic region's coordinates with the synthetic
    region's half_width_km (60 km) would run almost entirely off the edge
    of the real tile; load_real_polar_dem() clips safely when that happens,
    but the resulting crop is then off-center relative to what you asked
    for, which quietly breaks every downstream lat/lon in that crop. Use
    this function's returned center/half-width for the real-DEM run instead
    of regions.py's synthetic-tuned values, so the requested box is one the
    file actually, fully contains.
    """
    with rasterio.open(tif_path) as ds:
        src_crs = CRS.from_wkt(ds.crs.to_wkt())
        radius_m = src_crs.ellipsoid.semi_major_metre
        geo_crs = CRS.from_proj4(f"+proj=longlat +R={radius_m} +no_defs")
        to_geo = Transformer.from_crs(src_crs, geo_crs, always_xy=True)

        native_mpp = abs(ds.transform.a)
        cx_mid = (ds.bounds.left + ds.bounds.right) / 2.0
        cy_mid = (ds.bounds.bottom + ds.bounds.top) / 2.0
        lon_mid, lat_mid = to_geo.transform(cx_mid, cy_mid)

        width_km = ds.width * native_mpp / 1000.0
        height_km = ds.height * native_mpp / 1000.0

    return {
        "lat_center": lat_mid,
        "lon_center": lon_mid,
        "width_km": width_km,
        "height_km": height_km,
        "safe_half_width_km": min(width_km, height_km) / 2.0 * 0.95,  # 5% margin
    }


def load_real_polar_dem(
    tif_path: str,
    lat_center: float,
    lon_center: float,
    half_width_km: float,
    target_pixels: int = 800,
) -> DemCrop:
    """
    Read a windowed, optionally-decimated crop from a real PGDA south-pole
    LOLA GeoTIFF, centered on (lat_center, lon_center), covering a
    (2*half_width_km) x (2*half_width_km) box.

    target_pixels controls the OUTPUT grid size (not the source resolution):
    the native file is 5 m/pixel, so a 120 km box is 24000x24000 pixels at
    full resolution -- far too large to hold in memory or plot sensibly.
    Reading with out_shape=(target_pixels, target_pixels) and
    Resampling.average asks GDAL to do the decimation efficiently on read,
    the same way a lower-resolution DEM product is normally derived from a
    higher-resolution one.
    """
    with rasterio.open(tif_path) as ds:
        src_crs = CRS.from_wkt(ds.crs.to_wkt())  # rasterio.CRS -> pyproj.CRS (exposes .ellipsoid)
        radius_m = src_crs.ellipsoid.semi_major_metre
        geo_crs = CRS.from_proj4(f"+proj=longlat +R={radius_m} +no_defs")
        to_proj = Transformer.from_crs(geo_crs, src_crs, always_xy=True)
        to_geo = Transformer.from_crs(src_crs, geo_crs, always_xy=True)

        cx, cy = to_proj.transform(lon_center, lat_center)
        row, col = rasterio.transform.rowcol(ds.transform, cx, cy)

        native_mpp = abs(ds.transform.a)
        half_px = int(round(half_width_km * 1000.0 / native_mpp))
        row0, col0 = row - half_px, col - half_px
        width_px = height_px = 2 * half_px

        row0_clamped = max(row0, 0)
        col0_clamped = max(col0, 0)
        width_clamped = min(col0 + width_px, ds.width) - col0_clamped
        height_clamped = min(row0 + height_px, ds.height) - row0_clamped
        if width_clamped <= 0 or height_clamped <= 0:
            raise ValueError(
                f"Requested {half_width_km} km half-width around "
                f"({lat_center}, {lon_center}) falls entirely outside this "
                f"raster's coverage -- check the site file actually covers "
                f"this location, or use a smaller half_width_km."
            )
        window = Window(col0_clamped, row0_clamped, width_clamped, height_clamped)

        out_px = min(target_pixels, width_clamped, height_clamped)
        elevation_m = ds.read(
            1,
            window=window,
            out_shape=(out_px, out_px),
            resampling=Resampling.average,
        ).astype(np.float64)

        nodata = ds.nodata
        if nodata is not None:
            elevation_m = np.where(np.isclose(elevation_m, nodata), np.nan, elevation_m)

        effective_mpp = native_mpp * (width_clamped / out_px)

        # IMPORTANT: compute lat/lon bounds from the window that was ACTUALLY
        # read (row0_clamped/col0_clamped/width_clamped/height_clamped), not
        # from the originally-requested (unclamped) half_px box. PGDA's
        # per-site 5 m/pixel tiles are themselves small, fixed footprints
        # (e.g. Site23 is only ~21x21 km), so a requested half_width_km that
        # doesn't fully fit is common, not an edge case -- using the
        # unclamped corners here would silently mislabel every downstream
        # lat/lon (catalog-crater windowing, detection lat/lon, plotting)
        # with a box far larger than the data actually covers.
        row_top, row_bot = row0_clamped, row0_clamped + height_clamped
        col_left, col_right = col0_clamped, col0_clamped + width_clamped
        x_left, y_top = ds.transform * (col_left, row_top)
        x_right, y_bot = ds.transform * (col_right, row_bot)
        lon_a, lat_a = to_geo.transform(x_left, y_top)
        lon_b, lat_b = to_geo.transform(x_right, y_bot)

        was_clamped = (
            row0 != row0_clamped
            or col0 != col0_clamped
            or width_clamped != width_px
            or height_clamped != height_px
        )
        if was_clamped:
            print(
                f"  [real_polar_dem] WARNING: requested {half_width_km:.1f} km "
                f"half-width around ({lat_center}, {lon_center}) did not fully "
                f"fit inside this raster (it covers only {ds.width * native_mpp / 1000:.1f} x "
                f"{ds.height * native_mpp / 1000:.1f} km total) -- the returned "
                f"crop was clipped to what's actually available, and its "
                f"reported bounds reflect that clipped area, not the original request."
            )

    return DemCrop(
        elevation_m=elevation_m,
        lat_center=lat_center,
        lon_center=lon_center,
        half_width_deg=float("nan"),
        lat_bounds=(min(lat_a, lat_b), max(lat_a, lat_b)),
        lon_bounds=(min(lon_a, lon_b), max(lon_a, lon_b)),
        mpp_ns=effective_mpp,
        mpp_ew=effective_mpp,
        is_local_km=True,
    )
