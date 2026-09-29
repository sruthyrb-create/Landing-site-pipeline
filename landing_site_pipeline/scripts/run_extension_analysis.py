"""
Extension analysis: TOPSIS comparison + one-at-a-time weight-sensitivity
analysis, for both case-study regions. Builds on the exact same layers
run_pipeline.py computes -- this script does not re-derive crater
detections, slope, roughness, or illumination; it only adds a second way
to combine them (TOPSIS) and tests how much the weighted-sum result
depends on the chosen weights (OAT sensitivity).

Usage:
    PYTHONPATH=src python3 scripts/run_extension_analysis.py
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

from landing_site import craters, data, illumination, regions, scoring, sensitivity, synthetic, terrain, viz
from landing_site.sensitivity import _site_distance_km

DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)


def _prepare_tycho():
    dem = data.GlobalDem(DATA_DIR / "LunarLROLrocKaguya_1180mperpix_downsamp.png")
    catalog = data.load_head_catalog(DATA_DIR / "HeadCraters.csv")
    r = regions.REGION_TYCHO
    crop = dem.crop(r.lat_center, r.lon_center, r.half_width_deg)
    catalog_window = data.craters_in_window(catalog, crop.lat_bounds, crop.lon_bounds)
    dets = craters.detect_craters(crop)
    tm = terrain.compute_terrain_metrics(crop)
    illum = illumination.mid_latitude_incidence(crop)
    density = craters.crater_density_map(crop, dets, grid_cells=32)
    return crop, dets, catalog_window, tm, illum, density


def _prepare_malapert():
    catalog = data.load_head_catalog(DATA_DIR / "HeadCraters.csv")
    r = regions.REGION_MALAPERT

    def box_deg(margin_km):
        lat_m = margin_km / (np.pi * 1737.4 / 180.0)
        lon_m = margin_km / (np.pi * 1737.4 / 180.0 * np.cos(np.radians(r.lat_center)))
        return (r.lat_center - lat_m, r.lat_center + lat_m), (r.lon_center - lon_m, r.lon_center + lon_m)

    lat_b, lon_b = box_deg(r.half_width_km * 1.6)
    catalog_for_terrain = data.craters_in_window(catalog, lat_b, lon_b)
    lat_b_tight, lon_b_tight = box_deg(r.half_width_km)
    catalog_window = data.craters_in_window(catalog, lat_b_tight, lon_b_tight)

    crop = synthetic.build_synthetic_dem(
        r.lat_center, r.lon_center, r.half_width_km,
        craters_window=catalog_for_terrain,
        massif_height_km=5.0, massif_radius_km=12.0,
    )
    dets = craters.detect_craters(crop, min_radius_km=3.0, max_radius_km=40.0, threshold=0.03)
    tm = terrain.compute_terrain_metrics(crop)
    illum = illumination.polar_illumination_proxy(crop)
    density = craters.crater_density_map(crop, dets, grid_cells=32)
    return crop, dets, catalog_window, tm, illum, density


def analyze_region(label, crop, dets, catalog_window, tm, illum, density, out_prefix):
    print(f"\n=== Extension analysis: {label} ===")

    # --- Weighted-sum baseline (same as scoring.compute_suitability defaults) ---
    weighted_suit = scoring.compute_suitability(density, tm.slope_deg, illum)
    weighted_best = scoring.best_site(weighted_suit, crop)
    print(f"  Weighted-sum best site: lat={weighted_best['lat']:.3f}, lon={weighted_best['lon']:.3f}, "
          f"score={weighted_best['score']:.1f}/100")

    # --- TOPSIS on the identical three layers ---
    topsis_suit = sensitivity.compute_topsis(density, tm.slope_deg, illum)
    topsis_best = scoring.best_site(topsis_suit, crop)
    print(f"  TOPSIS best site:       lat={topsis_best['lat']:.3f}, lon={topsis_best['lon']:.3f}, "
          f"score={topsis_best['score']:.1f}/100")

    agreement_km = _site_distance_km(
        crop, weighted_best["lat"], weighted_best["lon"], topsis_best["lat"], topsis_best["lon"]
    )
    print(f"  Distance between the two methods' chosen best site: {agreement_km:.1f} km")

    # --- One-at-a-time sensitivity analysis of the weighted-sum weights ---
    baseline_best, sens_results = sensitivity.one_at_a_time_sensitivity(density, tm.slope_deg, illum, crop)
    max_shift_km = max(float(np.max(r.shift_km)) for r in sens_results)
    max_score_swing = max(float(np.max(r.best_scores) - np.min(r.best_scores)) for r in sens_results)
    print(f"  OAT sensitivity: largest best-site shift across all weight sweeps = {max_shift_km:.1f} km; "
          f"largest best-score swing = {max_score_swing:.1f} points")

    # --- Figure 1: weighted-sum vs TOPSIS best site, overlaid on the suitability map ---
    fig, ax = plt.subplots(figsize=(7, 6))
    viz.plot_layer(weighted_suit, crop, f"{label}: weighted-sum vs TOPSIS best site",
                    cmap="viridis", cbar_label="suitability / closeness (0-100)", ax=ax, vmin=0, vmax=100)
    if crop.is_local_km:
        from landing_site.synthetic import latlon_to_local_km
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

    # --- Figure 2: OAT sensitivity (2 rows x 3 cols: score sweep, then shift sweep) ---
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
    crop_a, dets_a, cw_a, tm_a, illum_a, density_a = _prepare_tycho()
    result_a = analyze_region("Region A: Tycho crater", crop_a, dets_a, cw_a, tm_a, illum_a, density_a,
                               "region_a_tycho")

    crop_b, dets_b, cw_b, tm_b, illum_b, density_b = _prepare_malapert()
    result_b = analyze_region("Region B: Malapert Massif", crop_b, dets_b, cw_b, tm_b, illum_b, density_b,
                               "region_b_malapert")

    with open(OUT_DIR / "extension_analysis_summary.txt", "w") as f:
        f.write("Extension Analysis -- TOPSIS Comparison + OAT Weight-Sensitivity Summary\n")
        f.write("=" * 74 + "\n\n")
        for label, res in [("Region A: Tycho", result_a), ("Region B: Malapert Massif", result_b)]:
            f.write(f"{label}\n")
            f.write(f"  Weighted-sum best site: lat={res['weighted_best']['lat']:.3f}, "
                    f"lon={res['weighted_best']['lon']:.3f}, score={res['weighted_best']['score']:.1f}/100\n")
            f.write(f"  TOPSIS best site:       lat={res['topsis_best']['lat']:.3f}, "
                    f"lon={res['topsis_best']['lon']:.3f}, score={res['topsis_best']['score']:.1f}/100\n")
            f.write(f"  Agreement between methods: {res['agreement_km']:.1f} km apart\n")
            f.write(f"  OAT sensitivity: max best-site shift = {res['max_shift_km']:.1f} km, "
                    f"max best-score swing = {res['max_score_swing']:.1f} points\n\n")
    print("\nSaved outputs/extension_analysis_summary.txt")
