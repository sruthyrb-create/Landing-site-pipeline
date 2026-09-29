"""
Runs the full pipeline on Region B (Malapert Massif) using the REAL NASA GSFC
PGDA 5 m/pixel south-pole LOLA DEM, instead of the synthetic terrain in
synthetic.py. This is a separate script from run_pipeline.py (which keeps
using the synthetic terrain) so the original, already-verified results stay
reproducible even after you add the real data.

Before running this, download the real DEM (see README's "Using the real
PGDA DEM for Malapert" section for the exact download link and instructions)
and set REAL_DEM_PATH below to point at the file you downloaded.

Usage:
    PYTHONPATH=src python3 scripts/run_real_malapert.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from landing_site import craters, data, illumination, real_polar_dem, regions, scoring, terrain, viz
from landing_site.data import MOON_RADIUS_M

DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# EDIT THIS to wherever you saved the downloaded file, e.g.:
#   REAL_DEM_PATH = DATA_DIR / "Site23_final_adj_5mpp_surf.tif"
REAL_DEM_PATH = DATA_DIR / "Site23_final_adj_5mpp_surf.tif"


def run_real_malapert():
    if not REAL_DEM_PATH.exists():
        raise FileNotFoundError(
            f"Real DEM not found at {REAL_DEM_PATH}.\n"
            f"Download it from https://pgda.gsfc.nasa.gov/data/LOLA_5mpp/Site23/"
            f"Site23_final_adj_5mpp_surf.tif and save it into {DATA_DIR}/, "
            f"or edit REAL_DEM_PATH at the top of this script to point at it."
        )

    print("\n=== Region B: Malapert Massif (REAL NASA PGDA 5 m/pixel south-pole LOLA DEM) ===")
    catalog = data.load_head_catalog(DATA_DIR / "HeadCraters.csv")
    r = regions.REGION_MALAPERT

    # Don't reuse regions.REGION_MALAPERT's (lat_center, lon_center,
    # half_width_km) here -- those describe the SYNTHETIC stand-in terrain
    # (a 120x120 km illustrative box), not this specific downloaded PGDA
    # tile. Ask the real file for its own true footprint instead, so the
    # crop we request is one the file actually, fully contains -- PGDA's
    # per-site tiles are small, fixed footprints (Site23 here is only
    # ~21x21 km), not an open-ended mosaic you can crop anywhere within.
    footprint = real_polar_dem.raster_footprint_latlon(str(REAL_DEM_PATH))
    km_per_deg_lat = np.pi * MOON_RADIUS_M / 1000.0 / 180.0  # ~30.3 km/deg on the Moon (not Earth's 111)
    lat_offset_km = (footprint["lat_center"] - r.lat_center) * km_per_deg_lat
    print(
        f"  Real tile footprint: {footprint['width_km']:.1f} x {footprint['height_km']:.1f} km, "
        f"centered at lat={footprint['lat_center']:.3f}, lon={footprint['lon_center']:.3f}"
    )
    print(
        f"  (regions.py's synthetic Malapert center is lat={r.lat_center:.3f}, lon={r.lon_center:.3f} "
        f"-- about {abs(lat_offset_km):.0f} km away in latitude alone. The synthetic terrain was built "
        f"as an illustrative stand-in, not surveyed to this exact tile's footprint, so the tile's own "
        f"center below is the correct choice for the real-data run.)"
    )

    crop = real_polar_dem.load_real_polar_dem(
        str(REAL_DEM_PATH),
        lat_center=footprint["lat_center"],
        lon_center=footprint["lon_center"],
        half_width_km=footprint["safe_half_width_km"],
        target_pixels=800,
    )
    print(f"  Loaded real DEM crop: {crop.shape[0]}x{crop.shape[1]} px, "
          f"{crop.extent_km()[0]:.1f} x {crop.extent_km()[1]:.1f} km, "
          f"effective resolution {crop.mpp_ew:.1f} m/pixel")

    catalog_window = data.craters_in_window(catalog, crop.lat_bounds, crop.lon_bounds)
    print(f"  Real catalog craters (>=20km) inside this window: {len(catalog_window)}")

    dets = craters.detect_craters(crop, min_radius_km=3.0, max_radius_km=40.0, threshold=0.03)
    val = craters.validate_against_catalog(dets, catalog_window, match_radius_km=10.0)
    print(f"  CV detections on real terrain: {len(dets.lat)} | matched to real catalog craters: "
          f"{val.n_matched}/{val.n_catalog_in_window}")

    tm = terrain.compute_terrain_metrics(crop)
    illum = illumination.polar_illumination_proxy(crop)
    density = craters.crater_density_map(crop, dets, grid_cells=32)

    suitability = scoring.compute_suitability(density, tm.slope_deg, illum)
    best = scoring.best_site(suitability, crop)
    print(f"  Best-scoring site: lat={best['lat']:.3f}, lon={best['lon']:.3f}, score={best['score']:.1f}/100")

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    viz.plot_dem_with_craters(crop, dets, catalog_window, "Malapert region: REAL PGDA DEM + real craters", ax=axes[0, 0])
    viz.plot_layer(tm.slope_deg, crop, "Slope (deg) [real terrain]", "inferno", "degrees", ax=axes[0, 1])
    viz.plot_layer(tm.roughness_m, crop, "Local roughness [real terrain]", "inferno", "meters (std-dev)", ax=axes[0, 2])
    viz.plot_layer(illum, crop, "Illumination proxy (height-above-median)", "cividis", "score 0-1", ax=axes[1, 0], vmin=0, vmax=1)
    viz.plot_layer(density, crop, "Crater density (coarse grid)", "magma", "weighted count", ax=axes[1, 1])
    viz.plot_suitability(suitability, crop, best=best, title="Composite suitability score", ax=axes[1, 2])
    fig.suptitle(
        "Region B -- Malapert Massif (REAL NASA GSFC PGDA 5 m/pixel LOLA DEM, real Head et al. 2010 catalog)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "region_b_malapert_REAL_full_pipeline.png", dpi=130)
    plt.close(fig)
    print(f"  saved outputs/region_b_malapert_REAL_full_pipeline.png")

    with open(OUT_DIR / "real_malapert_summary.txt", "w") as f:
        f.write("Region B: Malapert Massif -- REAL PGDA DEM run\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Effective resolution: {crop.mpp_ew:.1f} m/pixel (native file is 5 m/pixel; "
                f"decimated on read to keep the array a manageable size)\n")
        f.write(f"Detections: {len(dets.lat)}\n")
        f.write(f"Catalog craters (>=20km) in window: {val.n_catalog_in_window}\n")
        f.write(f"Matched (recall): {val.n_matched}/{val.n_catalog_in_window} "
                f"({val.recall_ge20km:.0%})\n")
        f.write(f"Best site: lat={best['lat']:.3f}, lon={best['lon']:.3f}, score={best['score']:.1f}/100\n")
    print(f"  saved outputs/real_malapert_summary.txt")

    return {"crop": crop, "detections": dets, "validation": val, "best": best}


if __name__ == "__main__":
    run_real_malapert()
