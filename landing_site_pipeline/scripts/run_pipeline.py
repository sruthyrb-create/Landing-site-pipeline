"""
End-to-end driver: runs the full landing-site suitability pipeline for both
case-study regions and writes all figures + a text summary to outputs/.

Usage:
    PYTHONPATH=src python3 scripts/run_pipeline.py
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

from landing_site import craters, data, illumination, regions, scoring, synthetic, terrain, viz

DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)


def run_tycho():
    print("\n=== Region A: Tycho crater (real DEM + real catalog) ===")
    dem = data.GlobalDem(DATA_DIR / "LunarLROLrocKaguya_1180mperpix_downsamp.png")
    catalog = data.load_head_catalog(DATA_DIR / "HeadCraters.csv")
    r = regions.REGION_TYCHO

    crop = dem.crop(r.lat_center, r.lon_center, r.half_width_deg)
    catalog_window = data.craters_in_window(catalog, crop.lat_bounds, crop.lon_bounds)

    dets = craters.detect_craters(crop)
    val = craters.validate_against_catalog(dets, catalog_window)
    print(f"  {crop.shape[0]}x{crop.shape[1]} px crop, extent {crop.extent_km()[0]:.0f} x {crop.extent_km()[1]:.0f} km")
    print(f"  CV detections: {len(dets.lat)} | catalog craters (>=20km) in window: {val.n_catalog_in_window}")
    print(f"  Matched: {val.n_matched}/{val.n_catalog_in_window} (recall={val.recall_ge20km:.0%}) | "
          f"uncatalogued candidates (<20km or unmatched): {val.n_uncatalogued_candidates}")

    tm = terrain.compute_terrain_metrics(crop)
    illum = illumination.mid_latitude_incidence(crop)
    density = craters.crater_density_map(crop, dets, grid_cells=32)

    suitability = scoring.compute_suitability(density, tm.slope_deg, illum)
    best = scoring.best_site(suitability, crop)
    print(f"  Best-scoring site: lat={best['lat']:.3f}, lon={best['lon']:.3f}, score={best['score']:.1f}/100")

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    viz.plot_dem_with_craters(crop, dets, catalog_window, "Tycho region: DEM + crater detections", ax=axes[0, 0])
    viz.plot_layer(tm.slope_deg, crop, "Slope (deg)", "inferno", "degrees", ax=axes[0, 1])
    viz.plot_layer(tm.roughness_m, crop, "Local roughness", "inferno", "meters (std-dev)", ax=axes[0, 2])
    viz.plot_layer(illum, crop, "Illumination (cosine-incidence model)", "cividis", "score 0-1", ax=axes[1, 0], vmin=0, vmax=1)
    viz.plot_layer(density, crop, "Crater density (coarse grid)", "magma", "weighted count", ax=axes[1, 1])
    viz.plot_suitability(suitability, crop, best=best, title="Composite suitability score", ax=axes[1, 2])
    fig.suptitle("Region A -- Tycho crater (real LOLA/Kaguya DEM, real Head et al. 2010 catalog)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "region_a_tycho_full_pipeline.png", dpi=130)
    plt.close(fig)
    print(f"  saved outputs/region_a_tycho_full_pipeline.png")

    return {"region": r, "detections": dets, "validation": val, "best": best}


def run_malapert():
    print("\n=== Region B: Malapert Massif (synthetic terrain informed by real craters) ===")
    catalog = data.load_head_catalog(DATA_DIR / "HeadCraters.csv")
    r = regions.REGION_MALAPERT

    def box_deg(margin_km):
        lat_m = margin_km / (np.pi * 1737.4 / 180.0)
        lon_m = margin_km / (np.pi * 1737.4 / 180.0 * np.cos(np.radians(r.lat_center)))
        return (r.lat_center - lat_m, r.lat_center + lat_m), (r.lon_center - lon_m, r.lon_center + lon_m)

    # A crater centered just outside the plotted box can still have its rim
    # poke into it, so the terrain builder gets a margined query...
    lat_b, lon_b = box_deg(r.half_width_km * 1.6)
    catalog_for_terrain = data.craters_in_window(catalog, lat_b, lon_b)
    # ...but validation/scoring only counts catalog craters actually inside
    # the analyzed box itself.
    lat_b_tight, lon_b_tight = box_deg(r.half_width_km)
    catalog_window = data.craters_in_window(catalog, lat_b_tight, lon_b_tight)
    print(f"  Real catalog craters (>=20km) inside the {2*r.half_width_km:.0f} km box: {len(catalog_window)}")

    crop = synthetic.build_synthetic_dem(
        r.lat_center, r.lon_center, r.half_width_km,
        craters_window=catalog_for_terrain,
        massif_height_km=5.0, massif_radius_km=12.0,
    )

    dets = craters.detect_craters(crop, min_radius_km=3.0, max_radius_km=40.0, threshold=0.03)
    val = craters.validate_against_catalog(dets, catalog_window, match_radius_km=10.0)
    print(f"  CV detections on synthetic terrain: {len(dets.lat)} | matched to real catalog craters: {val.n_matched}/{val.n_catalog_in_window}")

    tm = terrain.compute_terrain_metrics(crop)
    illum = illumination.polar_illumination_proxy(crop)
    density = craters.crater_density_map(crop, dets, grid_cells=32)

    suitability = scoring.compute_suitability(density, tm.slope_deg, illum)
    best = scoring.best_site(suitability, crop)
    print(f"  Best-scoring site: lat={best['lat']:.3f}, lon={best['lon']:.3f}, score={best['score']:.1f}/100")

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    viz.plot_dem_with_craters(crop, dets, catalog_window, "Malapert region: SYNTHETIC terrain + real craters", ax=axes[0, 0])
    viz.plot_layer(tm.slope_deg, crop, "Slope (deg) [synthetic terrain]", "inferno", "degrees", ax=axes[0, 1])
    viz.plot_layer(tm.roughness_m, crop, "Local roughness [synthetic terrain]", "inferno", "meters (std-dev)", ax=axes[0, 2])
    viz.plot_layer(illum, crop, "Illumination proxy (height-above-median)", "cividis", "score 0-1", ax=axes[1, 0], vmin=0, vmax=1)
    viz.plot_layer(density, crop, "Crater density (coarse grid)", "magma", "weighted count", ax=axes[1, 1])
    viz.plot_suitability(suitability, crop, best=best, title="Composite suitability score", ax=axes[1, 2])
    fig.suptitle("Region B -- Malapert Massif (SYNTHETIC terrain; real crater catalog; illustrative illumination proxy)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "region_b_malapert_full_pipeline.png", dpi=130)
    plt.close(fig)
    print(f"  saved outputs/region_b_malapert_full_pipeline.png")

    return {"region": r, "detections": dets, "validation": val, "best": best}


if __name__ == "__main__":
    result_a = run_tycho()
    result_b = run_malapert()

    with open(OUT_DIR / "summary.txt", "w") as f:
        f.write("Landing-Site Suitability Pipeline -- Summary\n")
        f.write("=" * 50 + "\n\n")
        for label, res in [("Region A: Tycho (real DEM + real catalog)", result_a),
                            ("Region B: Malapert Massif (synthetic + real craters)", result_b)]:
            f.write(f"{label}\n")
            f.write(f"  Detections: {len(res['detections'].lat)}\n")
            f.write(f"  Catalog craters (>=20km) in window: {res['validation'].n_catalog_in_window}\n")
            f.write(f"  Matched (recall): {res['validation'].n_matched}/{res['validation'].n_catalog_in_window} "
                    f"({res['validation'].recall_ge20km:.0%})\n")
            f.write(f"  Best site: lat={res['best']['lat']:.3f}, lon={res['best']['lon']:.3f}, "
                    f"score={res['best']['score']:.1f}/100\n\n")
    print(f"\nSaved outputs/summary.txt")
