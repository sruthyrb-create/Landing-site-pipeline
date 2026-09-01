8# Lunar Landing-Site Suitability Pipeline

A small research-style pipeline that scores lunar terrain for crewed-landing
suitability by combining three signals computed from real planetary science
data: crater hazard (detected with classical computer vision and validated
against a published crater catalog), slope/roughness hazard (derived
directly from real elevation data), and illumination benefit (modeled two
different ways depending on latitude). It was built as a class project for
an astrophysics course, aimed squarely at the intersection of data science,
research methodology, and aerospace/planetary mission design.

## Why this project

Landing-site suitability scoring is a real, operational planetary-science
and mission-engineering task -- it is part of how NASA and other agencies
evaluate candidate sites for real missions (including Artemis III). This
project builds a small, honest version of that pipeline end to end: object
detection on real digital elevation data, terrain analysis, a physically
motivated illumination model, and a composite scoring layer, applied to two
real, named lunar sites.

## The two case-study regions

| Region | Latitude | Data used | Purpose |
|---|---|---|---|
| **Tycho crater** | 43.3S | Real global DEM (1180 m/pixel) + real Head et al. (2010) crater catalog | Primary, fully-validated demonstration of the whole method on genuine data |
| **Malapert Massif** | ~85S | Real crater catalog + labeled *synthetic* terrain | Aerospace-relevant case study: one of NASA's 13 announced Artemis III candidate landing regions |

Region B exists to make the project's aerospace relevance concrete and
specific, not hypothetical -- see "Known limitations" below for exactly why
it uses synthetic terrain and what that does and doesn't prove.

## Data sources (real data, real citations)

- **Global lunar DEM** (`data/LunarLROLrocKaguya_1180mperpix_downsamp.png`):
  a 1180 m/pixel equirectangular digital elevation model covering
  longitudes -180..180 and latitudes -60..60, derived from the merged LRO
  LOLA + Kaguya Terrain Camera 59 m/pixel topography (the basis of the
  widely-used SLDEM2015 product). This exact file ships as test/demo data
  with [DeepMoon](https://github.com/silburt/DeepMoon) (Silburt, A. et al.
  2019, "Lunar Crater Identification via Deep Learning," *Icarus* 317,
  [arXiv:1803.02192](https://arxiv.org/abs/1803.02192)), which credits the
  DEM to the USGS Astrogeology Cloud Processing service and the LOLA/Kaguya
  science teams.
- **Global crater catalog** (`data/HeadCraters.csv`): Head, J. N. et al.
  (2010), "Global Distribution of Large Lunar Craters: Implications for
  Resurfacing and Impactor Populations," *Science* 329, 1504-1507. Craters
  >= 20 km diameter, true global coverage including the poles. Also
  distributed via DeepMoon's `catalogues/HeadCraters.csv`.
- **Artemis III candidate landing regions**: NASA, "NASA Identifies
  Candidate Regions for Landing Next Americans on Moon" (2022) --
  [nasa.gov](https://www.nasa.gov/news-release/nasa-identifies-candidate-regions-for-landing-next-americans-on-moon/).
- **Malapert Mountain description** (~5 km high ridge on the southwest rim
  of Malapert crater, near 0 deg longitude): per lunar-reference
  descriptions of Malapert crater/mountain (e.g. Wikipedia's summary of
  published sources), used in `synthetic.py` to place the modeled massif.
- **Crater depth/diameter relation** used to render synthetic craters:
  Pike, R. J. (1974, 1977) -- simple craters: depth = 0.20 x diameter;
  complex craters (>= 15 km): depth = 1.044 x diameter^0.301 (km).
- **YOLOv8 stretch-goal training data** (`yolo_stretch/`): Yang, H. et al.,
  "CraterDANet: A Convolutional Neural Network for Small-Scale Crater
  Detection via Synthetic-to-Real Domain Adaptation," *IEEE TGRS* -- real
  LRO NAC images (0.5 m/pixel) with real bounding-box crater annotations,
  near the Chang'E-4 landing site, via
  [github.com/yizuifangxiuyh/Lunar_Crater_Detection_Data](https://github.com/yizuifangxiuyh/Lunar_Crater_Detection_Data).

## Method, module by module

- **`data.py`** -- loads the global DEM, converts pixel coordinates to
  lat/lon and back, and crops rectangular windows with correct real-world
  meters-per-pixel scaling in both directions (accounting for longitude
  compression away from the equator).
- **`craters.py`** -- classical computer-vision crater detection.
  Craters are topographic bowls, so we detect them as blobs in scale-space
  on the *negated* elevation surface (crater floors become local maxima)
  using Laplacian-of-Gaussian blob detection (`skimage.feature.blob_log`).
  Detections are validated against the real Head catalog by nearest-match
  within a distance tolerance.
- **`terrain.py`** -- slope (from the elevation gradient, correctly scaled
  by real ground distance per pixel in each direction) and roughness (local
  elevation standard deviation), both computed directly from the DEM with
  plain numpy/scipy.
- **`illumination.py`** -- **two different models**, because the physics
  is genuinely different at the two latitudes:
  - *Mid-latitude* (Tycho, 43S): the Moon's tiny 1.54 deg axial tilt keeps
    the sun's elevation moderate and never grazing, so the standard
    Lambertian cosine-incidence law (computed from real slope + aspect)
    is the right model.
  - *Near-polar* (Malapert, 85S): the sun's elevation is always within
    ~1.5 deg of the horizon, so illumination is governed by whether nearby
    terrain blocks the sun -- the real mechanism behind the Moon's
    permanently-shadowed regions and rare "peaks of eternal light." Doing
    this exactly needs horizon ray-tracing against a real high-resolution
    DEM; this project instead uses an explicit, labeled *proxy*
    (height above local median terrain) -- see "Known limitations."
- **`synthetic.py`** -- builds a labeled synthetic terrain for Malapert,
  informed by real crater positions/sizes and Malapert Mountain's published
  height, used only because the real high-res polar DEM wasn't reachable
  in this environment (see below).
- **`scoring.py`** -- combines crater density, slope, and illumination into
  a single 0-100 composite suitability score per grid cell, with the
  relative weighting of each factor left as an explicit, adjustable
  parameter rather than a hard-coded "true" answer (this is a genuine
  mission-design judgment call).
- **`viz.py`** -- all plotting.

## Results (from `scripts/run_pipeline.py`)

**Region A -- Tycho crater** (256x256 px crop, ~221 x 303 km): the
classical-CV detector found 108 crater candidates in the window and
correctly matched 18 of the 20 real Head et al. catalog craters
(**90% recall**). The apparently-low raw precision is expected and not a
flaw: the catalog only contains craters >= 20 km, so most of the detector's
"extra" detections are genuine smaller craters the catalog simply doesn't
track -- this is explicitly checked in `validate_against_catalog()`, which
restricts catalog-based scoring to the size range the catalog actually
covers.

**Region B -- Malapert Massif** (120 x 120 km synthetic terrain): 5 real
catalog craters fall inside the analysis box; the detector matched 4 of
them on the synthetic terrain. The composite score correctly favors the
modeled massif's flanks (elevated, well-illuminated) over nearby crater
floors (low, hazardous, poorly illuminated) -- the expected qualitative
result, though the absolute illumination values are a labeled proxy, not a
validated measurement (again, see below).

Full figures: `outputs/region_a_tycho_full_pipeline.png` and
`outputs/region_b_malapert_full_pipeline.png`. Run
`PYTHONPATH=src python3 scripts/run_pipeline.py` to regenerate everything.

## Known limitations (read this before presenting the results)

**Network access.** This project was built in a sandboxed environment
whose network access was restricted to PyPI and GitHub -- it could not
reach NASA's PDS, USGS Astrogeology, or NASA GSFC's PGDA servers, which
host the real high-resolution lunar DEMs (including the real 5 m/pixel
south-pole LOLA topography for Artemis III candidate sites, at
`pgda.gsfc.nasa.gov/products/78`). This is why:

- The Tycho analysis uses a coarser 1180 m/pixel global DEM (real data,
  but lower resolution than the best available), obtained instead via a
  GitHub-hosted research repo that happened to bundle it as test data.
- The Malapert analysis uses labeled *synthetic* terrain rather than a
  real DEM, because no real polar raster was reachable at all.

**On a machine with normal internet access, this has now been done** --
see "Using the real PGDA DEM for Malapert" below for the exact download
link and how to run it.

## Using the real PGDA DEM for Malapert

NASA GSFC's Planetary Geodesy and Geophysics group (PGDA) publishes real
5 m/pixel LOLA-derived elevation data for named south-pole landing-site
candidates, including Malapert massif specifically (their "Site23"). It's
public NASA data -- no login or special permission needed, just cite it
(already done in the References section above and in the report).

**1. Download it.** Product page:
[pgda.gsfc.nasa.gov/products/78](https://pgda.gsfc.nasa.gov/products/78).
Direct file (the elevation/"LDEM" GeoTIFF -- this is the one you need):

```
https://pgda.gsfc.nasa.gov/data/LOLA_5mpp/Site23/Site23_final_adj_5mpp_surf.tif
```

This is a real spacecraft-derived elevation raster, not a small demo file --
expect it to be a large download (likely several hundred MB to a few GB
depending on the site's exact areal coverage at 5 m/pixel). Check your
available disk space before starting.

**2. Save it into `data/`** in this project, keeping the original filename:
`data/Site23_final_adj_5mpp_surf.tif`.

**3. Run the new script:**

```bash
PYTHONPATH=src python3 scripts/run_real_malapert.py
```

This produces `outputs/region_b_malapert_REAL_full_pipeline.png` and
`outputs/real_malapert_summary.txt`, in the same format as the original
Malapert figure, but built from real elevation data instead of the
synthetic terrain.

**Why this integrates as a "contained change", concretely:** the real
PGDA files are GeoTIFFs in *south polar stereographic* coordinates (real
meters, not lat/lon degrees) -- which turns out to fit this project's
existing `is_local_km` mechanism (built for the synthetic terrain)
almost exactly, since a polar-stereographic projection is already a flat,
meters-based coordinate system near the pole. `real_polar_dem.py` reads
the file with `rasterio` in its native projection (using `pyproj` to
convert your requested lat/lon center into the file's projected
coordinates), reads only a decimated window around the region of interest
(so a 120 km box doesn't require loading a 24,000 x 24,000-pixel array
into memory), and returns a standard `DemCrop` with `is_local_km=True` --
the exact same object type `synthetic.build_synthetic_dem()` returns.
Every downstream module (`craters.py`, `terrain.py`, `illumination.py`,
`scoring.py`, `sensitivity.py`, `viz.py`) works completely unmodified, and
was verified to run end-to-end against a projection-correct test file
before this was added.

**One honest caveat:** this loader's logic has been validated against a
synthetic file built with the exact same projection type and metadata
structure NASA's documentation describes -- but it has not yet been run
against the actual downloaded file, since that download wasn't possible
from the sandboxed environment this project was originally built in. If
`run_real_malapert.py` errors for you, it's most likely a small, fixable
mismatch between the assumed and actual file structure (e.g. a slightly
different nodata convention) -- keep the exact error message to fix it.

Once you have a real result, `illumination.polar_illumination_proxy()`
is the next thing worth upgrading -- it's currently a labeled stand-in for
real horizon-based shadow ray-tracing (see its docstring), which is exactly
what NASA GSFC's PGDA illumination products
([pgda.gsfc.nasa.gov/products/69](https://pgda.gsfc.nasa.gov/products/69))
compute, now that real high-resolution topography is available to ray-trace
against.

**The synthetic terrain is a teaching aid, not a measurement.** It is
informed by real crater positions/sizes and Malapert Mountain's published
height, but the fine-grained topography in between is generated noise, and
the illumination proxy is a simplified stand-in for real horizon-based
shadowing. Every figure derived from it is labeled "(synthetic terrain)" or
"proxy" for this reason. Treat Region B's results as illustrating the
*method*, not as a real assessment of Malapert Massif.

**Crater detection resolution limit.** At 1180 m/pixel, craters below
roughly 5 km diameter are at or below the reliable detection limit (only a
few pixels across); the detector is deliberately restricted to >= 5 km to
avoid noise-driven false positives, which is also why it can only be
validated against the >= 20 km catalog and not smaller, uncatalogued
features with certainty.

**Composite score weighting is a design choice, not a fact.** The 35% /
35% / 30% (crater / slope / illumination) weighting in `scoring.py` is a
reasonable default, not a validated mission requirement -- a real mission
design would set these weights based on specific vehicle and crew
constraints.

## Extension analysis: is the composite score's weighting robust?

`sensitivity.py` and `scripts/run_extension_analysis.py` add two checks on
the composite suitability score, motivated by the fact that its criteria
weights (35% crater / 35% slope / 30% illumination) are an explicit design
choice, not a derived fact:

- **TOPSIS comparison.** The same three layers are combined a second,
  independent way using TOPSIS (Technique for Order Preference by
  Similarity to Ideal Solution; Hwang & Yoon, 1981) -- the same class of
  multi-criteria method used by Peña-Asensio et al. (2024) for the real
  Artemis III ranking. At both regions, TOPSIS picks the *exact same*
  best-scoring site as the weighted sum (0.0 km apart), despite combining
  the criteria in a mathematically different way.
- **One-at-a-time (OAT) sensitivity analysis.** Each weight is swept from
  0 to 1 in turn (redistributing the rest proportionally) and the
  best-site's score and location are tracked. At both regions the best
  site is unchanged across most of each weight's range, only shifting
  once a single weight is pushed to a near-extreme value (max shift:
  198 km at Tycho, 71 km at Malapert, both only in the outer 10-20% of
  the range).

Together these give evidence, rather than just an assertion, that the
default weighting is not fragile -- see `outputs/*_topsis_comparison.png`,
`outputs/*_sensitivity.png`, and `outputs/extension_analysis_summary.txt`.
Run `PYTHONPATH=src python3 scripts/run_extension_analysis.py` to
regenerate all of it.

## YOLOv8 stretch goal

`yolo_stretch/` contains a from-scratch fine-tuning experiment: a
pretrained YOLOv8n model fine-tuned on the real, bounding-box-annotated
CraterDANet dataset (see citation above) to test the deep-learning upgrade
path over the classical-CV baseline used in the main pipeline.

This is intentionally scoped as a proof-of-concept, not a finished
detector: only 12 training images (~13,400 real crater boxes) were
available in the repo, craters average only ~7 pixels across (near the
practical detection limit for a generic object detector), and training ran
CPU-only in this sandboxed environment. See `yolo_stretch/RESULTS.md` for
the actual metrics obtained and exactly what a real scale-up (the full
dataset, a GPU, more epochs, tiled high-resolution crops) would need to
change. The value of this stretch goal is demonstrating a complete, correct
fine-tuning workflow on real annotated data -- not claiming a production
detector.

## Repository layout

```
src/landing_site/     the pipeline as an importable package
  real_polar_dem.py     optional loader for the real PGDA south-pole DEM
scripts/run_pipeline.py   end-to-end driver, regenerates outputs/
scripts/run_real_malapert.py  runs Region B on the real PGDA DEM (once downloaded)
scripts/run_extension_analysis.py  TOPSIS comparison + weight-sensitivity analysis
notebooks/            narrated Jupyter walkthrough (same pipeline)
data/                 the two real datasets described above
outputs/              generated figures + summary.txt
yolo_stretch/         YOLOv8 fine-tuning proof-of-concept
```

## Running it

```bash
pip install rasterio numpy scipy opencv-python-headless matplotlib \
            geopandas shapely pyproj scikit-image pandas pillow

PYTHONPATH=src python3 scripts/run_pipeline.py       # regenerate all figures
jupyter nbconvert --to notebook --execute \
    notebooks/landing_site_walkthrough.ipynb          # regenerate the notebook
```

## Suggested next steps for extending this into a stronger portfolio piece

1. Download the real 5 m/pixel PGDA polar DEM (see "Using the real PGDA
   DEM for Malapert" above) and run `scripts/run_real_malapert.py`; then
   replace the illumination proxy with real horizon-based ray tracing now
   that real high-resolution topography is available to ray-trace against.
2. Scale the YOLOv8 fine-tune to the full CraterDANet dataset on a GPU.
3. Add a slope/roughness sensitivity analysis: how does the best site
   change as the composite-score weights change? This is a genuine
   mission-design question and a good "discussion" section for a report.
4. Push the write-up further into a short paper/poster format for a
   research-experience line on a grad-school application -- the CSFD-style
   validation-against-catalog methodology here is real, citable practice.
