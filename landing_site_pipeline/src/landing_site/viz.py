"""Plotting helpers -- kept separate from analysis code on purpose."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .data import DemCrop
from .synthetic import latlon_to_local_km


def _extent(dem_crop: DemCrop):
    if dem_crop.is_local_km:
        h, w = dem_crop.shape
        half_x = w * dem_crop.mpp_ew / 2000.0
        half_y = h * dem_crop.mpp_ns / 2000.0
        return [-half_x, half_x, -half_y, half_y]
    south, north = dem_crop.lat_bounds
    west, east = dem_crop.lon_bounds
    return [west, east, south, north]


def _axis_labels(ax, dem_crop: DemCrop):
    if dem_crop.is_local_km:
        ax.set_xlabel("Local east-west distance (km)")
        ax.set_ylabel("Local north-south distance (km)")
    else:
        ax.set_xlabel("Longitude (deg)")
        ax.set_ylabel("Latitude (deg)")


def _detection_xy(detections, dem_crop: DemCrop):
    """Detections in the plot's native coordinate system (km or lon/lat)."""
    if not len(getattr(detections, "lat", [])):
        return [], []
    if dem_crop.is_local_km:
        h, w = dem_crop.shape
        x_km = (np.asarray(detections.x_px) - w / 2) * dem_crop.mpp_ew / 1000.0
        y_km = (h / 2 - np.asarray(detections.y_px)) * dem_crop.mpp_ns / 1000.0
        return x_km, y_km
    return detections.lon, detections.lat


def _catalog_xy(catalog_window, dem_crop: DemCrop):
    if catalog_window is None or not len(catalog_window):
        return [], []
    if dem_crop.is_local_km:
        x_km, y_km = latlon_to_local_km(
            dem_crop.lat_center, dem_crop.lon_center,
            catalog_window["lat"].to_numpy(), catalog_window["lon"].to_numpy(),
        )
        return x_km, y_km
    return catalog_window["lon"], catalog_window["lat"]


def plot_dem_with_craters(dem_crop: DemCrop, detections, catalog_window, title, ax=None):
    ax = ax or plt.gca()
    ax.imshow(dem_crop.elevation_m, cmap="gray", extent=_extent(dem_crop), origin="upper", aspect="equal")

    dx, dy = _detection_xy(detections, dem_crop)
    if len(dx):
        ax.scatter(dx, dy, s=10, facecolors="none", edgecolors="lime", linewidths=1.0, label="CV detections")

    cx, cy = _catalog_xy(catalog_window, dem_crop)
    if len(cx):
        ax.scatter(cx, cy, s=25, marker="x", c="red", linewidths=1.2, label="Head et al. catalog (>=20km)")

    _axis_labels(ax, dem_crop)
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.6)
    return ax


def plot_layer(arr: np.ndarray, dem_crop: DemCrop, title, cmap, cbar_label, ax=None, vmin=None, vmax=None):
    ax = ax or plt.gca()
    im = ax.imshow(arr, cmap=cmap, extent=_extent(dem_crop), origin="upper", vmin=vmin, vmax=vmax, aspect="equal")
    plt.colorbar(im, ax=ax, label=cbar_label, fraction=0.046, pad=0.04)
    _axis_labels(ax, dem_crop)
    ax.set_title(title)
    return ax


def plot_suitability(suitability: np.ndarray, dem_crop: DemCrop, best=None, title="Suitability score", ax=None):
    ax = plot_layer(suitability, dem_crop, title, cmap="viridis", cbar_label="suitability (0-100)", ax=ax, vmin=0, vmax=100)
    if best is not None:
        if dem_crop.is_local_km:
            bx, by = latlon_to_local_km(dem_crop.lat_center, dem_crop.lon_center, best["lat"], best["lon"])
        else:
            bx, by = best["lon"], best["lat"]
        ax.scatter([bx], [by], marker="*", s=250, c="red", edgecolors="white", linewidths=1.2, label=f"best site ({best['score']:.0f})")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.6)
    return ax
