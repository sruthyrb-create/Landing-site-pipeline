"""
Real lunar elevation data: loading, coordinate transforms, and cropping.

Data source
-----------
LunarLROLrocKaguya_1180mperpix_downsamp.png
    A global equirectangular (Plate Carree) digital elevation model of the
    Moon at 1180 m/pixel, covering longitudes -180..180 deg and latitudes
    -60..60 deg. It is an 8-bit rescaling of a 16-bit GeoTIFF that was itself
    downsampled from the LRO LOLA + Kaguya Terrain Camera merged 59 m/pixel
    DEM (the basis of the widely-used SLDEM2015 product). Original elevation
    range: -21138 m to +21138 m (relative to the mean lunar radius of
    1737.4 km), linearly rescaled to 0-255.

    This exact file ships as test/demo data with the DeepMoon project
    (Silburt et al. 2019, "Lunar Crater Identification via Deep Learning",
    Icarus 317, https://arxiv.org/abs/1803.02192,
    https://github.com/silburt/DeepMoon), which in turn credits the DEM to
    the USGS Astrogeology Cloud Processing service and the LOLA/Kaguya
    science teams.

HeadCraters.csv
    Head, J. N. et al. (2010), "Global Distribution of Large Lunar Craters:
    Implications for Resurfacing and Impactor Populations", Science 329,
    1504-1507. Craters >= 20 km diameter, global coverage including the
    polar regions (unlike the DEM above, which stops at +/-60 deg latitude).
    Distributed alongside DeepMoon's `catalogues/HeadCraters.csv`.

Why +/-60 degrees only
-----------------------
The merged LOLA+Kaguya DEM this pipeline uses only extends to +/-60 degrees
latitude, so it cannot cover an actual lunar-south-pole Artemis candidate
site directly. `regions.py` documents how the pipeline is designed to be
pointed at a real, much higher-resolution polar DEM (NASA GSFC PGDA's 5 m/pixel
LOLA topography for south-pole landing sites) once that ~GB-scale file is
available on a machine with unrestricted internet access -- this sandboxed
environment could only reach PyPI and GitHub, not the PDS/PGDA data servers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

MOON_RADIUS_M = 1_737_400.0

# Geographic extent of the global DEM PNG (degrees).
LON_MIN, LON_MAX = -180.0, 180.0
LAT_MIN, LAT_MAX = -60.0, 60.0

# Original elevation range (meters) before it was rescaled to 8-bit.
ELEV_MIN_M, ELEV_MAX_M = -21138.0, 21138.0


@dataclass
class DemCrop:
    """A rectangular crop of the global DEM plus its geographic metadata."""

    elevation_m: np.ndarray        # (rows, cols) float64, meters
    lat_center: float
    lon_center: float
    half_width_deg: float
    lat_bounds: tuple[float, float]   # (south, north)
    lon_bounds: tuple[float, float]   # (west, east)
    mpp_ns: float                      # meters/pixel, north-south direction
    mpp_ew: float                      # meters/pixel, east-west direction at lat_center
    is_local_km: bool = False          # True for synthetic near-pole crops (see synthetic.py)

    @property
    def shape(self):
        return self.elevation_m.shape

    def extent_km(self):
        h, w = self.shape
        return (w * self.mpp_ew / 1000.0, h * self.mpp_ns / 1000.0)


class GlobalDem:
    """Wraps the global equirectangular PNG DEM and provides lat/lon cropping."""

    def __init__(self, png_path: str | Path):
        img = Image.open(png_path)
        self._pixels = np.asarray(img).astype(np.float64)
        self.height, self.width = self._pixels.shape
        self._ppd_x = self.width / (LON_MAX - LON_MIN)
        self._ppd_y = self.height / (LAT_MAX - LAT_MIN)

    def _elevation_m(self, pixel_values: np.ndarray) -> np.ndarray:
        return pixel_values / 255.0 * (ELEV_MAX_M - ELEV_MIN_M) + ELEV_MIN_M

    def latlon_to_px(self, lat: float, lon: float) -> tuple[float, float]:
        """Convert (lat, lon) in degrees to (x, y) pixel coordinates."""
        x = (lon - LON_MIN) * self._ppd_x
        y = (LAT_MAX - lat) * self._ppd_y  # row 0 = north edge (+60 deg)
        return x, y

    def crop(self, lat_center: float, lon_center: float, half_width_deg: float) -> DemCrop:
        """Crop a square-in-degrees window centered on (lat_center, lon_center)."""
        if not (LAT_MIN <= lat_center - half_width_deg and lat_center + half_width_deg <= LAT_MAX):
            raise ValueError(
                f"Requested window lat range "
                f"[{lat_center - half_width_deg}, {lat_center + half_width_deg}] "
                f"exceeds this DEM's coverage of [{LAT_MIN}, {LAT_MAX}] degrees."
            )
        x0, y0 = self.latlon_to_px(lat_center + half_width_deg, lon_center - half_width_deg)
        x1, y1 = self.latlon_to_px(lat_center - half_width_deg, lon_center + half_width_deg)
        xmin, xmax = sorted((int(round(x0)), int(round(x1))))
        ymin, ymax = sorted((int(round(y0)), int(round(y1))))
        xmin, ymin = max(xmin, 0), max(ymin, 0)
        xmax, ymax = min(xmax, self.width), min(ymax, self.height)

        pixel_crop = self._pixels[ymin:ymax, xmin:xmax]
        elevation_m = self._elevation_m(pixel_crop)

        lat_south = lat_center - half_width_deg
        lat_north = lat_center + half_width_deg
        lon_west = lon_center - half_width_deg
        lon_east = lon_center + half_width_deg

        mpp_ns = (np.pi * MOON_RADIUS_M * (lat_north - lat_south) / 180.0) / elevation_m.shape[0]
        mpp_ew = (
            2 * np.pi * MOON_RADIUS_M * np.cos(np.radians(lat_center))
            * (lon_east - lon_west) / 360.0
        ) / elevation_m.shape[1]

        return DemCrop(
            elevation_m=elevation_m,
            lat_center=lat_center,
            lon_center=lon_center,
            half_width_deg=half_width_deg,
            lat_bounds=(lat_south, lat_north),
            lon_bounds=(lon_west, lon_east),
            mpp_ns=mpp_ns,
            mpp_ew=mpp_ew,
        )


def load_head_catalog(csv_path: str | Path):
    """Load the Head et al. (2010) >=20 km global lunar crater catalog."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    df = df.rename(columns={"Lon": "lon", "Lat": "lat", "Diam_km": "diam_km"})
    return df


def craters_in_window(catalog_df, lat_bounds, lon_bounds):
    """Subset a crater catalog to a lat/lon window (handles simple ranges only)."""
    south, north = lat_bounds
    west, east = lon_bounds
    mask = (
        (catalog_df["lat"] >= south)
        & (catalog_df["lat"] <= north)
        & (catalog_df["lon"] >= west)
        & (catalog_df["lon"] <= east)
    )
    return catalog_df.loc[mask].copy()
