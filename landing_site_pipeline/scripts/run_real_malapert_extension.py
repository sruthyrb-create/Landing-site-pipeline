"""
Extension analysis (TOPSIS comparison + one-at-a-time weight-sensitivity)
run against the REAL NASA GSFC PGDA 5 m/pixel south-pole LOLA DEM for
Malapert Massif (Site23), instead of the earlier synthetic-terrain
prototype. Mirrors run_extension_analysis.py's Region B logic exactly,
but sourced from real_polar_dem.load_real_polar_dem() (see
run_real_malapert.py) rather than synthetic.build_synthetic_dem().

Usage:
    PYTHONPATH=src python3 scripts/run_real_malapert_extension.py
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

from landing_site import craters, data, illumination, real_polar_dem, regions, scoring, sensitivity, terrain, viz
from landing_site.sensitivity import _site_distance_km
from landing_site.synthetic import latlon_to_local_km

DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

REAL_DEM_PATH = DATA_DIR / "Site23_final_adj_5mpp_surf.tif"


def _prepare_real_malapert():
    catalog = data.load_head_catalog(DATA_DIR / "HeadCraters.csv")
    footprint = real_polar_dem.raster_footprint_latlon(str(REAL_DEM_PATH))
    crop = real_polar_dem.load_real_polar_dem(
        str(REAL_DEM_PATH),
        lat_center=footprint["lat_center"],
        lon_center=footprint["lon_center"],
        half_width_km=footprint["safe_half_width_km"],
        target_pixels=800,
    )
    catalog_window = data.craters_in_window(catalog, crop.lat_bounds, crop.lon_bounds)
    dets = craters.detect_craters(crop, min_radius_km=3.0, max_radius_km=40.0, threshold=0.03)
    tm = terrain.compute_terrain_metrics(crop)
    illum = illumination.polar_illumination_proxy(crop)
    density = craters.crater_density_map(crop, dets, grid_cells=32)
    return crop, dets, catalog_window, tm, illum, density


def analyze_region(label, crop, dets, catalog_window, tm, illum, density, out_prefix):
    print(f"\n=== Extension analysis: {label} ===")

    weighted_suit = scoring.compute_suitability(density, tm.slope_deg, illum)
    weighted_best = scoring.best_site(weighted_suit, crop)
    print(f"  Weighted-sum best site: lat={weighted_best['lat']:.3f}, lon={weighted_best['lon']:.3f}, "
          f"score={weighted_best['score']:.1f}/100")

    topsis_suit = sensitivity.compute_topsis(density, tm.slope_deg, illum)
    topsis_best = scoring.best_site(topsis_suit, crop)
    print(f"  TOPSIS best site:       lat={topsis_best['lat']:.3f}, lon={topsis_best['lon']:.3f}, "
          f"score={topsis_best['score']:.1f}/100")

    agreement_km = _site_distance_km(
        crop, weighted_best["lat"], weighted_best["lon"], topsis_best["lat"], topsis_best["lon"]
    )
    print(f"  Distance between the two methods' chosen best site: {agreement_km:.1f} km")

    baseline_best, sens_results = sensitivity.one_at_a_time_sensitivity(density, tm.slope_deg, illum, crop)
    max_shift_km = max(float(np.max(r.shift_km)) for r in sens_results)
    max_score_swing = max(float(np.max(r.best_scores) - np.min(r.best_scores)) for r in sens_results)
    print(f"  OAT sensitivity: largest best-site shift across all weight sweeps = {max_shift_km:.1f} km; "
          f"largest best-score swing = {max_score_swing:.1f} points")

    fig, ax = plt.subplots(figsize=(7, 6))
    viz.plot_layer(weighted_suit, crop, f"{label}: weighted-sum vs TOPSIS best site",
                    cmap="viridis", cbar_label="suitability / closeness (0-100)", ax=ax, vmin=0, vmax=100)
    if crop.is_local_km:
        wx, wy = latlon_to_local_km(crop.lat_center, crop.lon_center, weighted_best["lat"], weighted_best["lon"])
        tx, ty = latlon_to_local_km(crop.lat_center, crop.lon_center, topsis_best["lat"], topsis_best["lon"])
    else:
        wx, wy = weighted_best["lon"], weighted_best["lat"]
        tx, ty = topsis_best["lon"], topsis_best["lat"]
    ax.scatter([wx], [wy], marker="*", s=280, c="red", edgecolors="white", linewidths=1.2,
               label=f"weighted-sum best ({weighted_best['score']:.0f})")
    ax.scatter([tx], [ty], marker="D", s=110, c="gold", edgecolors="black", linewidths=1.0,
               label=f"TOPSIS best ({topsis_best['score']:.0f})")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.7)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{out_prefix}_topsis_comparison.png", dpi=130)
    plt.close(fig)
    print(f"  saved outputs/{out_prefix}_topsis_comparison.png")

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for col, res in enumerate(sens_results):
        axes[0, col].plot(res.weight_values, res.best_scores, marker="o", color="#2b6cb0")
        axes[0, col].axvline(getattr(scoring.SuitabilityWeights().normalized(), res.varied_criterion),
                              color="gray", linestyle="--", linewidth=1, label="default weight")
        axes[0, col].set_title(f"Best score vs {res.varied_criterion} weight")
        axes[0, col].set_xlabel(f"w_{res.varied_criterion}")
        axes[0, col].set_ylabel("best-site score (0-100)")
        axes[0, col].legend(fontsize=7)

        axes[1, col].plot(res.weight_values, res.shift_km, marker="o", color="#c05621")
        axes[1, col].axvline(getattr(scoring.SuitabilityWeights().normalized(), res.varied_criterion),
                              color="gray", linestyle="--", linewidth=1)
        axes[1, col].set_title(f"Best-site shift vs {res.varied_criterion} weight")
        axes[1, col].set_xlabel(f"w_{res.varied_criterion}")
        axes[1, col].set_ylabel("distance from baseline best site (km)")
    fig.suptitle(f"{label}: one-at-a-time sensitivity of the composite suitability score", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{out_prefix}_sensitivity.png", dpi=130)
    plt.close(fig)
    print(f"  saved outputs/{out_prefix}_sensitivity.png")

    return {
        "weighted_best": weighted_best,
        "topsis_best": topsis_best,
        "agreement_km": agreement_km,
        "max_shift_km": max_shift_km,
        "max_score_swing": max_score_swing,
    }


if __name__ == "__main__":
    crop, dets, cw, tm, illum, density = _prepare_real_malapert()
    result = analyze_region(
        "Region B: Malapert Massif (REAL PGDA DEM)", crop, dets, cw, tm, illum, density,
        "region_b_malapert_REAL",
    )

    with open(OUT_DIR / "real_malapert_extension_summary.txt", "w") as f:
        f.write("Region B: Malapert Massif -- REAL PGDA DEM -- Extension Analysis\n")
        f.write("=" * 66 + "\n\n")
        f.write(f"Weighted-sum best site: lat={result['weighted_best']['lat']:.3f}, "
                f"lon={result['weighted_best']['lon']:.3f}, score={result['weighted_best']['score']:.1f}/100\n")
        f.write(f"TOPSIS best site:       lat={result['topsis_best']['lat']:.3f}, "
                f"lon={result['topsis_best']['lon']:.3f}, score={result['topsis_best']['score']:.1f}/100\n")
        f.write(f"Agreement between methods: {result['agreement_km']:.1f} km apart\n")
        f.write(f"OAT sensitivity: max best-site shift = {result['max_shift_km']:.1f} km, "
                f"max best-score swing = {result['max_score_swing']:.1f} points\n")
    print("\nSaved outputs/real_malapert_extension_summary.txt")
