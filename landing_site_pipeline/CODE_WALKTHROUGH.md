# Code Walkthrough — Landing-Site Suitability Pipeline

This document explains **every file** in the project, in plain language, so you can defend
any part of it during your evaluation. For each file it covers: what the file is for, what
every function does (line by line where it matters), and — most importantly — *why* it was
built that way, because "why did you do X instead of Y" is the question you are most likely
to get.

Read this alongside the actual code (`landing_site_pipeline/`). Line numbers below match the
files exactly, so if you're asked "show me where X happens," you can point straight to it.

**How to use this before your evaluation:** don't memorize this document word for word. Read
each section, then close it and try to explain that file out loud in your own words, using the
code open next to you. If you can explain *why* a design choice was made (not just *what* the
code does), you can handle almost any follow-up question.

---

## 1. Project layout, in one picture

```
src/landing_site/      <- the actual method, as a reusable Python package
    data.py            <- load real elevation data, crop it, load the crater catalog
    regions.py          <- the two case-study locations (Tycho, Malapert) and why we picked them
    craters.py          <- crater detection (computer vision) + scoring against real catalog
    terrain.py           <- slope and roughness
    illumination.py      <- two sunlight models (mid-latitude vs polar)
    synthetic.py          <- an early development-stage prototype terrain for the polar site (superseded, kept for history only)
    real_polar_dem.py      <- loads the real NASA GSFC PGDA 5 m/pixel south-pole LOLA DEM for Malapert
    scoring.py               <- combines everything into one suitability number
    sensitivity.py            <- TOPSIS comparison + one-at-a-time weight-sensitivity analysis
    viz.py                     <- all the plotting

scripts/run_pipeline.py             <- runs Tycho + Malapert (real DEM) end-to-end and saves the figures
scripts/run_real_malapert.py         <- focused driver for Malapert against the real PGDA DEM
scripts/run_extension_analysis.py     <- TOPSIS + sensitivity for Tycho
scripts/run_real_malapert_extension.py <- TOPSIS + sensitivity for Malapert, real PGDA DEM
notebooks/                          <- the same pipeline, narrated, in Jupyter
yolo_stretch/                        <- the optional deep-learning experiment (separate from the main pipeline)
```

The most important thing to be able to say in one sentence: **"Each module does one job, and
`run_pipeline.py` wires them together."** `data.py` never does plotting. `viz.py` never does
detection. This separation is itself a design decision worth defending — it's what makes the
code testable and reusable, and it is exactly the same reason the notebook and the script can
share the same underlying package instead of duplicating logic.

---

## 2. `src/landing_site/data.py` — loading real data

**Job:** load the real global elevation image, convert pixel coordinates to lat/lon and back,
and cut out a rectangular window ("crop") around a point of interest with the *correct*
real-world scale.

### Constants (lines 48–55)
```python
MOON_RADIUS_M = 1_737_400.0
LON_MIN, LON_MAX = -180.0, 180.0
LAT_MIN, LAT_MAX = -60.0, 60.0
ELEV_MIN_M, ELEV_MAX_M = -21138.0, 21138.0
```
The Moon's mean radius (used later for distance math), and the real geographic extent of the
global DEM file: it covers the whole Moon in longitude, but only ±60° in latitude — **this
single fact is the entire reason Region B (Malapert, ~86°S) needed a different data source**.
The fix now used is `real_polar_dem.py` (Section 8), which loads a dedicated real 5 m/pixel
polar DEM tile instead of this global ±60° file. An earlier, now-superseded prototype filled
this gap with generated terrain (`synthetic.py`, Section 7) before that real polar tile had
been downloaded and integrated.
`ELEV_MIN_M/MAX_M` are the true elevation range the 8-bit PNG was scaled from, so we can
convert pixel brightness (0–255) back into real meters.

### `DemCrop` (lines 58–78)
A plain data container — one crop of elevation data plus everything you need to interpret it:
its center, its real lat/lon bounds, and its meters-per-pixel scale **in both directions
separately** (`mpp_ns` vs `mpp_ew`). Every other module in the project takes a `DemCrop` as
input, which is why swapping data sources later (real high-res DEM instead of synthetic)
requires no changes anywhere else — *if you're asked how you'd extend this, this is the
answer: everything downstream only needs a `DemCrop`, so a new data source just needs a new
function that produces one.*

The `is_local_km` flag (default `False`) marks whether this crop uses lat/lon degrees as its
coordinate system (real regions) or local flat kilometers (the synthetic polar region) — more
on why in `regions.py` and `synthetic.py`.

### `GlobalDem.__init__` (lines 84–89)
Loads the PNG with PIL, converts it to a NumPy array of floats, and computes "pixels per
degree" in x and y from the image's actual dimensions divided by its known geographic extent.

### `_elevation_m` (lines 91–92)
```python
return pixel_values / 255.0 * (ELEV_MAX_M - ELEV_MIN_M) + ELEV_MIN_M
```
Converts an 8-bit pixel value (0–255) back to real elevation in meters. This is a simple
linear rescale — reversing exactly what was done to produce the 8-bit PNG in the first place.

### `latlon_to_px` (lines 94–98)
Converts a (latitude, longitude) into a pixel (x, y) position in the image. Note `y = (LAT_MAX
- lat) * ppd_y` — row 0 of the image is the *north* edge, so latitude decreases as you go down
the image, which is why it's `LAT_MAX - lat` and not `lat - LAT_MIN`.

### `crop` (lines 100–138)
This is the important one. Given a center point and a half-width in degrees:
1. **Checks the request is inside the DEM's real coverage** (line 102) and raises a clear
   error if not — this is what would immediately tell you "you can't crop Malapert this way,"
   which is exactly the discovery that led to building `synthetic.py`.
2. Converts the four corners to pixel coordinates and slices the array.
3. **Computes the real meters-per-pixel in each direction separately** (lines 123–127). This
   matters because on a sphere, a degree of longitude covers less real ground the further you
   are from the equator (it shrinks by a factor of `cos(latitude)`), while a degree of
   latitude covers roughly the same real distance everywhere. If you only had one "meters per
   pixel" number, your slope calculation in `terrain.py` would be wrong by that
   `cos(latitude)` factor. This is a real, physically-motivated detail, not an arbitrary
   complication.
4. Returns a `DemCrop` with all of that metadata attached.

### `load_head_catalog` and `craters_in_window` (lines 141–160)
Load the real Head et al. crater catalog CSV and rename its columns to a consistent naming
scheme; `craters_in_window` just filters the catalog rows down to whatever lat/lon box you're
currently analyzing, using simple boolean masking on a pandas DataFrame.

**Likely question: "Why not use a proper GIS library like rasterio for all this?"** Answer:
the DEM here is a plain PNG without embedded geographic metadata (no GeoTIFF headers), so
there is no georeferencing for a GIS library to read automatically — the lat/lon math has to
be done explicitly anyway, and doing it explicitly in plain NumPy keeps every step visible and
auditable rather than hidden inside a library call.

---

## 3. `src/landing_site/regions.py` — the two case studies

**Job:** define exactly where the pipeline points, and document, in the code itself, why those
two places were chosen.

### `Region` (lines 23–31)
A frozen (immutable) dataclass holding a name, center coordinates, a human-readable
description, and *either* `half_width_deg` (a degree-based box) *or* `half_width_km` (a
real-kilometer box) — never both at once for the same region.

### `REGION_TYCHO` (lines 34–48)
43.3°S, 11.2°W, `half_width_deg=5.0`. Chosen because it sits safely inside the DEM's ±60°
coverage and is a real, well-known, heavily cratered site — good for testing crater detection
against a real catalog.

### `REGION_MALAPERT` (lines 50–75)
~84.9°S, 0.4°E, `half_width_km=60.0` (a 120×120 km box). This is the aerospace-relevant site —
a real NASA Artemis III candidate region. Notice it uses `half_width_km`, **not**
`half_width_deg`. The docstring spells out the reason with a concrete number: *at 85°S, 4° of
longitude is only about 11 km of real ground, while 4° of latitude is about 121 km.* A
fixed-degree box that close to the pole would be a long, thin, badly distorted sliver rather
than a square region — this was an actual bug I hit early on (see the "bugs I hit" section
near the end of this document), and `half_width_km` plus the `is_local_km` machinery in
`data.py`/`real_polar_dem.py`/`viz.py` is the fix.

**Where these specific numbers (84.9°S, 0.4°E) came from, and why the real DEM disagrees with
them by ~33 km:** this coordinate was set early on to the catalogued center of *Malapert
crater* (84°54′S, 12°54′E per the crater's published catalog entry), on the reasonable-seeming
assumption that "Malapert" and "Malapert Massif" share one location. They don't: the massif is
a specific ridge on the crater's rim, not the crater's own center, and once the real NASA GSFC
PGDA 5 m/pixel LOLA tile for the massif (`Site23_final_adj_5mpp_surf.tif`, used by
`real_polar_dem.py`) was downloaded and its own embedded geographic metadata read directly, its
true center turned out to be about 85.995°S, −0.236°E — roughly 33 km away from the value
above. `run_real_malapert.py` no longer uses this module's `REGION_MALAPERT` coordinates for
Malapert at all; it reads the real tile's own footprint directly (see Section 8's
`raster_footprint_latlon`), which sidesteps the crater-vs-massif naming ambiguity entirely by
trusting the file's own metadata rather than a hand-typed coordinate. `REGION_MALAPERT` is kept
in this module only as the region originally used to build the early synthetic-terrain
prototype (Section 7) — it is not used by the current real-data path.

**Likely question: "Why not just use a proper map projection (like polar stereographic)?"**
Honest answer: a real mission-design tool would; for a 120×120 km box this small, the flat
local-kilometer approximation used here (see `synthetic.py`) introduces negligible distortion,
and it's far simpler to build and explain than a full projection library. This is a scope
trade-off, stated as one.

---

## 4. `src/landing_site/craters.py` — crater detection & validation

**Job:** find craters in elevation data (no machine learning, no training data needed), and
check the results against a real independent catalog.

### The core idea (module docstring, lines 1–23)
A crater is a bowl: a small region of *locally low* elevation surrounded by higher terrain (a
rim). Standard blob detectors look for **bright** blobs — regions of locally *high* intensity.
So the trick (line 70) is to invert the elevation before feeding it to the detector:
```python
inverted = 1.0 - e_norm
```
After this, crater floors (originally low points) become the *brightest* spots in the image,
which is exactly what a blob detector is built to find.

### `detect_craters` (lines 56–97)
1. **Normalize elevation to [0, 1] and invert it** (lines 68–70), as just described.
2. **Convert your desired min/max crater radius (in km) into "sigma" values** the blob
   detector understands (lines 72–74). The Laplacian-of-Gaussian detector works in a
   mathematical quantity called sigma (the width of a Gaussian blur kernel), not directly in
   physical radius, so there's a conversion: `radius = sigma * sqrt(2)`, rearranged to solve
   for sigma, and then converted from real kilometers into image pixels using the crop's
   metres-per-pixel value.
3. **Run `skimage.feature.blob_log`** (lines 76–82) — this is the actual detection call. It
   scans the image at many different Gaussian blur scales between `min_sigma` and `max_sigma`
   and reports every local maximum above `threshold` as a detected blob, together with which
   scale it was strongest at (which tells you its radius).
4. **Convert each detection back into real units** (lines 87–95): pixel radius to km using the
   metres-per-pixel scale, and pixel (x, y) to real lat/lon using the crop's known bounds — a
   simple linear interpolation across the image.

**Likely question: "Why min_radius_km=5.0 specifically?"** This was tuned, not guessed: an
earlier attempt with `min_radius_km=3.0` produced 221 detections, but 166 of them were pinned
right at the 3 km floor — a strong sign those were noise/pixel-level artifacts, since at 1180
m/pixel a 3 km crater is barely 2–3 pixels across, below where the detector can reliably tell a
real crater from image noise. Raising the floor to 5 km removed most of that noise while
barely changing recall (still 90%), which is documented in the README and Section 5.4 of the
report. **This is a real example of tuning a computer-vision parameter against evidence, not a
default value left untouched** — a good thing to be ready to explain.

### `validate_against_catalog` (lines 100–155)
1. **Restrict the catalog to craters ≥20 km** (line 112) — because that's genuinely all the
   Head catalog contains; comparing against catalog entries the catalog was never built to
   have would be meaningless.
2. **Greedy nearest-neighbor matching** (lines 118–136): for every detected crater, compute its
   distance in real kilometers to every catalog crater (using a small-angle approximation —
   good enough at the scale of one region, not meant to be used globally), and if the closest
   catalog crater is within `match_radius_km` (default 15 km) *and* hasn't already been claimed
   by an earlier detection, count it as a match.
3. **Compute precision and recall** (lines 144–145). Precision = matched detections ÷ total
   detections. Recall = matched catalog craters ÷ total catalog craters. **The report
   deliberately treats recall as the meaningful number** (Section 3.4.2/5.1) — because the
   catalog's 20 km size floor guarantees many genuine smaller-crater detections will have no
   catalog match, which would drag precision down artificially even if every single detection
   were a real crater.

### `crater_density_map` (lines 158–174)
Bins detected craters into a coarse grid (default 20×20 cells) and counts them per cell,
weighting bigger craters more heavily (`np.clip(radius_km / 5.0, 1.0, None)`, line 171) —
because a single large crater is a bigger hazard to a landing site than several tiny ones, so
it should count for more in the hazard map that feeds `scoring.py`.

---

## 5. `src/landing_site/terrain.py` — slope and roughness

**Job:** two simple, transparent terrain-hazard measurements computed directly from elevation.

### `compute_slope_deg` (lines 24–33)
```python
dz_dy, dz_dx = np.gradient(elevation_m, mpp_ns, mpp_ew)
slope_rad = np.arctan(np.hypot(dz_dx, dz_dy))
```
`np.gradient` estimates how fast elevation changes from one pixel to the next, in each
direction — and crucially, it's given the *real* meters-per-pixel spacing in each direction
(not just "1 pixel"), so the result is a real physical slope, not a pixel-based one.
`np.hypot(dz_dx, dz_dy)` combines the two directional slopes into a single steepest-slope
magnitude (Pythagoras), and `arctan` converts that ratio into an actual angle in degrees. This
is the same core method any terrain-analysis GIS tool uses; it's just written out explicitly
here instead of hidden inside a library call.

### `compute_roughness_m` (lines 36–55)
Local roughness = how "bumpy" the terrain is in a small neighborhood, measured as the standard
deviation of elevation within a small window (default 5×5 pixels). Instead of the slow way (a
sliding-window loop, or `scipy.ndimage.generic_filter` with a Python callable, which is
mentioned but not actually used), it uses a standard statistical identity:
```
variance = mean(x²) − mean(x)²
```
computed with two fast `uniform_filter` calls (lines 51–53) instead of one slow custom
filter — same mathematical result, much faster. This is a real, common performance trick worth
being able to explain if asked "why two `uniform_filter` calls instead of one obvious one?"

---

## 6. `src/landing_site/illumination.py` — two sunlight models

**Job:** estimate how well-lit a point is — but using **two genuinely different physical
models**, because the physics is genuinely different at the two latitudes analyzed.

### Why two models (module docstring, lines 1–38)
At Tycho (43°S), the Moon's tiny 1.54° axial tilt keeps the sun's elevation angle moderate
(never near the horizon), so which way a slope *faces* relative to the sun is what matters —
the standard "cosine law" from basic radiometry. Near the pole (Malapert, ~85°S), the sun's
elevation angle is always within about 1.5° of the horizon, so the dominant question isn't
which way a slope faces — it's whether *nearby terrain physically blocks the sun*. This is
the real mechanism behind the Moon's permanently-shadowed crater floors and its rare "peaks of
eternal light" (elevated points that stay above the local horizon almost all the time).

### `mid_latitude_incidence` (lines 55–77)
Computes real slope and aspect (downslope-facing direction) from the DEM, then applies the
standard Lambertian cosine-incidence formula:
```
cos(i) = sin(sun_elevation) * cos(slope) + cos(sun_elevation) * sin(slope) * cos(sun_azimuth − aspect)
```
This says: a flat surface (`slope=0`) gets illumination purely from the sun's elevation angle;
a tilted surface gets a bonus if it faces toward the sun (`aspect` close to `sun_azimuth`) and
a penalty if it faces away. This is exact physics, not an approximation — the only
simplification is using one fixed representative sun position rather than simulating a full
lunar day.

### `polar_illumination_proxy` (lines 80–96)
This is explicitly labeled a **proxy**, not a real illumination calculation, and this is one of
the most important things to be ready to explain honestly. The real calculation needs
horizon ray-tracing: for every point, check in every compass direction whether terrain blocks
the sun at every time of the lunar day, against a high-resolution DEM. That is a genuinely
different (and much heavier) computation from what this function does — it needs a full
sun-position/time-of-day simulation, not just a static elevation snapshot — so it was out of
scope for this project even now that the real 5 m/pixel south-pole PGDA/LOLA DEM (see
`real_polar_dem.py`, Section 8) has been downloaded and is used for every other Malapert
calculation. Instead, this function scores a point by *how much higher it is than the
surrounding median terrain*, passed through a sigmoid (S-curve):
```python
return 1.0 / (1.0 + np.exp(-relative_height_km / (height_scale_km / 4.0)))
```
This captures the right qualitative direction — taller points are more likely to clear the
local horizon and stay lit — without claiming to be a validated illumination fraction. **If
asked "is this physically accurate," the honest answer is no — it's a stand-in for the real
ray-tracing calculation, clearly labeled as such everywhere it's used, and this is true
regardless of whether the underlying elevation data is real or synthetic** — the proxy nature
of this specific function is about the *illumination model*, not about which DEM feeds it.

---

## 7. `src/landing_site/synthetic.py` — an early, now-superseded prototype

> **Status: historical only.** Everything in this section describes an early
> pipeline-development tool. It was used only while building and testing the analysis code,
> before the real NASA GSFC PGDA 5 m/pixel south-pole LOLA DEM tile for Malapert Massif had
> been downloaded and integrated (see `real_polar_dem.py`, Section 8). It does not contribute
> to any figure, table, or number reported in the final report or in Chapter 4's results —
> every current Malapert result comes from the real DEM. This module is kept in the codebase
> for transparency about how the project developed, not because it is part of the current
> method.

**Original job:** before real polar elevation data was on hand, this module built a *labeled*
substitute terrain for Malapert so that the crater-detection, scoring, and sensitivity code
could be written and tested end-to-end, grounded in two real facts rather than invented from
nothing.

### Why this existed (module docstring, lines 1–38)
Two real facts anchored the synthetic terrain:
1. The **real** positions and diameters of every Head et al. catalog crater near the
   coordinates then assumed for Malapert, rendered using a **real, published** depth/diameter
   relation (Pike, 1974/1977).
2. Malapert Mountain's **real, published** description as roughly a 5 km high ridge.

Everything else (the base rolling terrain in between) was generated smoothed random noise, and
this was stated explicitly and repeatedly in this module's own docstring: *"This is explicitly
a teaching aid, not a measurement."*

### `latlon_to_local_km` / `local_km_to_latlon` (lines 49–65)
A flat local-approximation coordinate conversion: treats a small patch of the Moon's surface
near the pole as if it were flat, converting lat/lon offsets into real east/north kilometer
offsets (and back) using simple trigonometry. This is valid for a small area but would break
down over a much larger one — a genuine, stated scope limitation. This helper function is the
one piece of this module still actively used: `real_polar_dem.py`, `sensitivity.py`, and the
run scripts all import it to convert real-DEM lat/lon into local kilometers for distance math
and plotting, since it's a generically useful small-area coordinate helper independent of
where the elevation values themselves came from.

### `_crater_depth_km` (lines 68–72)
Directly implements the two Pike equations from the theory chapter:
```python
simple = 0.20 * diam_km
complex_ = 1.044 * np.power(diam_km, 0.301)
return np.where(diam_km < 15.0, simple, complex_)
```
`np.where` picks the simple-crater formula for small craters and the complex-crater formula
for large ones, based on the real 15 km transition point from the literature.

### `build_synthetic_dem` (lines 75–141)
1. Built a base rolling-hills terrain from Gaussian-smoothed random noise, scaled to a
   realistic highland roughness amplitude (default 250 m) (lines 99–101).
2. Built real-kilometer x/y coordinate grids (lines 103–105) — note this is genuine physical
   distance, not degrees.
3. **For every real catalog crater in the window** (lines 109–119): converted its true lat/lon
   to local km, computed its real depth from `_crater_depth_km`, and stamped a smooth
   cosine-profile bowl into the terrain at that exact location and size.
4. **Added a raised dome for the massif itself** (lines 121–124) at a fixed offset, using a
   Gaussian bump shape.
5. Returned a `DemCrop` with `is_local_km=True` — the same flag `real_polar_dem.py` sets on its
   `DemCrop` output today, which is exactly why swapping this module out for the real DEM later
   required no changes anywhere else in the pipeline (`craters.py`, `terrain.py`,
   `illumination.py`, `scoring.py`, `viz.py` all just consume a `DemCrop`).

**Likely question: "Isn't this making up data?"** Be ready to answer this directly and calmly:
it was *not* pretending to be a measurement even at the time — every figure derived from it was
labeled "(synthetic terrain)," and only the real crater positions/sizes and the real massif
height were used as inputs. It has since been fully superseded: the current report and every
current Malapert number come from the real PGDA DEM (`real_polar_dem.py`), and this module now
exists purely as a record of how the pipeline was developed and tested before that real tile was
in hand — a good, honest answer if asked why the codebase still contains it.

---

## 8. `src/landing_site/real_polar_dem.py` — the real polar DEM for Malapert Massif

**Job:** load the real NASA GSFC PGDA (Planetary Geodesy and Geophysics group) 5 m/pixel
south-pole LOLA elevation tile for the Malapert Massif site
(`Site23_final_adj_5mpp_surf.tif`, a GeoTIFF in south polar stereographic projection), and
produce a `DemCrop` from it exactly like `data.py`'s `crop()` does for the mid-latitude global
DEM — so that every downstream module (`craters.py`, `terrain.py`, `illumination.py`,
`scoring.py`, `sensitivity.py`, `viz.py`) works on real Malapert data completely unchanged.

### Why a separate module instead of extending `data.py`
`data.py`'s `GlobalDem` only understands a full-Moon, plain-PNG, ±60°-latitude image with no
embedded coordinate system — pixel-to-lat/lon math is done by hand because there is no
georeferencing to read. The PGDA polar tile is the opposite: it's a **GeoTIFF** with real
embedded map projection metadata (south polar stereographic, read via `rasterio` and
`pyproj`), and it's a small, fixed-size site tile (a few thousand pixels across, covering only
about 21×21 km) rather than a crop-anywhere global mosaic. Those are different enough data
shapes that a second, dedicated loader was clearer than bending `GlobalDem` to do both.

### `raster_footprint_latlon(tif_path)`
Reads the GeoTIFF's own projected bounds and CRS (coordinate reference system) directly with
`rasterio`, converts its four corners from the projection's meters into real lat/lon using
`pyproj.Transformer`, and returns the tile's own true center latitude/longitude, its real
width/height in km, and a `safe_half_width_km` (a conservative half-width guaranteed to fit
inside the tile without hitting its edges). **This function is why the real-data pipeline no
longer needs anyone to type in a hand-guessed center coordinate for Malapert at all** —
`run_real_malapert.py` and `run_real_malapert_extension.py` both call this first and use its
answer directly, rather than trusting a coordinate written down separately in `regions.py`.
This sidesteps the crater-vs-massif naming mixup described in Section 3 entirely, because the
footprint comes from the file's own metadata rather than a hand-typed value.

### `load_real_polar_dem(tif_path, lat_center, lon_center, half_width_km, target_pixels)`
1. Converts the requested lat/lon center and half-width into the tile's projected (meters)
   coordinate system, then into a pixel row/column window, using the same `rasterio`/`pyproj`
   transforms as above.
2. **Clamps that window to the tile's actual pixel bounds** if the requested box would run off
   the edge of the real data (a small ~21×21 km tile can't always supply the full box a caller
   asks for), and prints a warning when clamping happened (`was_clamped`).
3. Reads the (possibly clamped) window of real elevation values, resamples it to
   `target_pixels` for a manageable array size, and — this is the part that was originally
   buggy (see the "bugs I found and fixed" section) — computes the crop's `lat_bounds`/
   `lon_bounds` from the **actual clamped pixel window that was read**, not from the original,
   unclamped request. Getting this right matters because every downstream lat/lon (catalog
   crater positions, detection positions, the best-site star, every axis label) is computed
   from these bounds — if they don't match the pixels actually loaded, everything drawn on top
   of the terrain silently lands in the wrong place relative to it, even though the terrain
   image itself looks fine.
4. Returns a `DemCrop` with `is_local_km=False` (it carries real lat/lon, unlike the old
   synthetic module's local-km crops) plus the real meters-per-pixel scale, so every
   downstream module treats it exactly like the Tycho crop.

**Likely question: "How is this different from just trusting `regions.py`'s coordinates?"**
`regions.py`'s `REGION_MALAPERT` value was a hand-typed estimate (originally Malapert crater's
catalogued center, not the massif's) that turned out to be about 33 km from the real tile's
true center. `raster_footprint_latlon` instead reads the coordinate directly out of the file's
own embedded map projection — it can't drift from what the data actually covers, because it
*is* what the data covers.

---

## 9. `src/landing_site/scoring.py` — combining everything into one score

**Job:** turn three separate layers (crater hazard, slope hazard, illumination benefit) into
one composite 0–100 suitability number per location.

### `_normalize01` (lines 28–32)
Min-max normalization: rescales any array so its lowest value becomes 0 and its highest
becomes 1, so that three layers with completely different natural units (crater counts,
degrees, a 0–1 illumination score) can be combined fairly.

### `SuitabilityWeights` (lines 35–43)
A small dataclass holding the three weights (crater=0.35, slope=0.35, illumination=0.30) with
a `.normalized()` method that rescales them to sum to exactly 1 even if you passed in weights
that didn't. **These specific numbers are a reasonable starting default, not a scientifically
derived answer** — the report says this explicitly (Section 3.4.5, 5.5) and compares against
Peña-Asensio et al. (2024), whose professional-grade study used seven criteria and a formal
method (TOPSIS) with a tested sensitivity analysis instead of a fixed weighted sum. Section 10
below covers `sensitivity.py`, which runs the equivalent checks on this project's own weights
rather than only pointing at someone else's.

### `compute_suitability` (lines 46–74)
1. **Resamples every input to a common grid size** (lines 60–65) using `scipy.ndimage.zoom` —
   necessary because `crater_density` is deliberately computed on a coarser grid (see
   `crater_density_map`) than the per-pixel slope and illumination layers.
2. Normalizes each resampled layer to [0, 1].
3. Combines them:
```python
suitability = 100 * (w.crater * (1 - crater_norm) + w.slope * (1 - slope_norm) + w.illumination * illum_norm)
```
Note the `(1 - ...)` on the two hazard terms — high crater density and high slope should
*lower* the score, so their normalized value is inverted before being weighted in, while
illumination is a benefit so it's added directly.

### `best_site` (lines 77–85)
Finds the single highest-scoring pixel (`np.nanargmax`) and converts its pixel position back
to a real lat/lon using the crop's known bounds — the same linear-interpolation idea used
elsewhere in the project.

---

## 10. `src/landing_site/sensitivity.py` — is the composite score's weighting actually trustworthy?

**Job:** this module doesn't compute any new physical layer — it exists purely to stress-test
`scoring.py`'s composite score, because a weighted-sum formula with hand-picked weights is
the single most subjective part of the whole pipeline, and asserting it's "reasonable" in a
docstring isn't the same as showing it. This is the module you should reach for if asked "how
do you know your weights weren't just picked to make the answer look good."

### `compute_topsis` (lines 88–139)
Runs the same three layers (`crater_density`, `slope_deg`, `illumination_score`) through a
**completely different** combination method: TOPSIS (Technique for Order Preference by
Similarity to Ideal Solution — a real, named, published method from Hwang & Yoon, 1981, and
the same general class of method Peña-Asensio et al. (2024) used for the real Artemis III
ranking). Instead of a weighted average, TOPSIS:
1. Vector-normalizes each criterion layer (line ~118) — a different normalization than
   `scoring.py`'s min-max normalization, on purpose, since the point is to test with a
   genuinely independent method, not a cosmetic variation of the same one.
2. Weights each normalized layer (same weights as the weighted-sum, for a fair comparison).
3. Finds the best-possible and worst-possible combination of all three criteria across the
   whole region (the "ideal best" and "ideal worst" points) — for crater/slope this is the
   minimum (lower hazard is better); for illumination it's the maximum (line ~127–134).
4. For every pixel, computes its distance to both the ideal best and ideal worst point, and
   scores it by `distance_to_worst / (distance_to_best + distance_to_worst)` — a pixel close
   to the ideal and far from the worst case scores near 1.

**The result:** at both Tycho and Malapert (Malapert run against the real PGDA DEM via
`run_real_malapert_extension.py`), TOPSIS's best-scoring pixel is the *exact same pixel* the
weighted sum picked — 0.0 km apart in both regions. That agreement is meaningful specifically
because the two methods were never designed to agree; if the weighted-sum result had really
just been an artifact of that one formula, a genuinely different method would have had every
opportunity to disagree, and it didn't.

### `one_at_a_time_sensitivity` (lines 156–199)
This answers a different question: not "does a different method agree," but "how much does
the *same* method's answer change if the weights are wrong?" For each of the three weights in
turn, it sweeps that one weight from 0 to 1 (lines 175–178 work out how to redistribute the
remaining budget between the other two, keeping their relative proportion fixed so all three
still sum to 1), recomputes the suitability map and its best site at every step (line 187), and
records how far that site is (in real km, via `_site_distance_km`) from the default-weight
best site.

**The result:** at both regions, the best site barely moves across most of each weight's range
— it only jumps once a single weight is pushed to a near-extreme value. At Tycho, the largest
shift recorded was 198 km, only in roughly the last 10–20% of that weight's range. At
Malapert, run against the real PGDA DEM, the largest best-site shift recorded across all three
weight sweeps was 17.6 km, with the largest best-score swing only 0.2 points — if anything a
*more* robust result than Tycho's, and one that (unlike the earlier synthetic-terrain version
of this test) is now based entirely on real elevation data. **This is the evidence behind the
report's claim that the 0.35/0.35/0.30 default isn't fragile** — a reasonable range of
alternative weightings around it would have picked essentially the same site at both regions.

### `_site_distance_km` (lines 40–51)
A small helper that measures real distance between two sites correctly regardless of which
coordinate system the region uses — the same `is_local_km` branching idea from `viz.py`, reused
here so this module works unmodified on both regions. In practice, both Tycho and (now)
Malapert use real lat/lon (`is_local_km=False`), so this branching is mostly a legacy of the
module's design rather than something actively exercised today — but it's what let the
synthetic-terrain prototype and the real polar DEM share this code unchanged while the project
transitioned from one to the other.

**Likely question: "Doesn't this just prove the method agrees with itself?"** No — that's
exactly why two different techniques are used here rather than one. TOPSIS is not a variant of
the weighted sum; it doesn't use weights the same way and doesn't normalize the same way. Two
structurally different methods landing on the identical answer is stronger evidence than either
method alone.

**Likely question: "What does OAT sensitivity analysis NOT tell you?"** Be ready to say this
plainly — it only varies one weight at a time, so it can miss an effect that only shows up when
two weights change together. The Future Work section of the report names the fix (sampling the
full weight simplex, e.g. with random Dirichlet-distributed weight triples) as a natural next
step, which is an honest, specific answer rather than a vague "future work could improve this."

---

## 11. `src/landing_site/viz.py` — plotting

**Job:** every plot in the project, kept deliberately separate from the analysis code, so the
scientific logic (`craters.py`, `terrain.py`, etc.) never depends on matplotlib at all.

### The `is_local_km` branching (lines 12–53)
Nearly every function here checks `dem_crop.is_local_km` and does one of two things: for
regions using real lat/lon directly (Tycho, and now Malapert via the real PGDA DEM), plot
using longitude/latitude axes; for any crop flagged `is_local_km=True` (only the now-superseded
synthetic prototype ever set this for Malapert), convert everything to local kilometer
coordinates first (using `latlon_to_local_km`, imported from `synthetic.py`) and plot in
kilometers instead. This branching was the direct fix for the polar-distortion bug described in
`regions.py` — **without it, a polar region plotted on a plain degree-based axis would look
like a thin, squashed sliver instead of a proper square region**, because a real-kilometer-square
area exaggerates east-west distances enormously that close to the pole on a degree axis. The
real polar DEM's crop keeps `is_local_km=False` because `real_polar_dem.py` derives true lat/lon
bounds directly from the GeoTIFF's own projection, so this specific branch isn't exercised for
current Malapert plots, but the machinery is kept because it's exactly what let the project
switch data sources for Malapert without touching any plotting code.

### The plotting functions themselves
`plot_dem_with_craters` shows the elevation image in grayscale with detected craters (green
circles) and catalog craters (red X marks) overlaid. `plot_layer` is a generic single-layer
heatmap plotter reused for slope, roughness, illumination, and density. `plot_suitability`
reuses `plot_layer` and then adds a red star at the best-scoring site.

---

## 12. `scripts/run_pipeline.py` — the end-to-end driver

**Job:** the one script that ties every module above together and reproduces every number and
figure in the report. If asked "how do I know these results are real and not made up," this
script — plus the fact that the notebook re-executes the identical logic and gets identical
numbers — is the answer.

### `run_tycho` (lines 30–67)
In order: load the real DEM and catalog → crop the Tycho region → run crater detection →
validate against the catalog → compute slope/roughness → compute illumination → compute crater
density → combine into a suitability score → find the best site → build a 2×3 grid of plots →
save the figure and print the summary numbers that appear in the report's Table 4.1.

A sibling script, `scripts/run_extension_analysis.py`, reuses the exact same
`_prepare_tycho()`/`_prepare_malapert()` data-loading steps and then hands the resulting layers
to `sensitivity.py` (Section 10) instead of stopping at `scoring.compute_suitability` — it's
kept as a separate script rather than folded into `run_pipeline.py` so the original,
already-validated pipeline stays untouched and easy to point to on its own.

### `run_malapert` (lines 70–121)
Same overall shape, with one extra wrinkle worth being able to explain: the **margin trick**
(lines 80–88). The terrain-building step needs craters from a *slightly larger* box than the
one actually being analyzed and scored (`margin_km * 1.6`), because a real crater whose center
sits just outside the 120×120 km analysis box can still have its rim poking into the box — if
you only queried the tight box, you'd get an abrupt, unrealistic cutoff right at the edge where
a crater's bowl is sliced off mid-rim. Validation and scoring, on the other hand, only count
catalog craters that are genuinely inside the box being evaluated, so they use the tight query
instead. This distinction — "build with margin, score without it" — is a subtle but real
design decision, and a good one to mention if asked about edge cases.

### The `__main__` block (lines 124–141)
Runs both regions and writes `outputs/summary.txt` with the exact headline numbers (detections,
recall, best site, score) for both regions — this file is the source of the numbers quoted
throughout the report and README.

### `scripts/run_real_malapert.py` — the focused real-data driver for Malapert
This is the script that replaced the old synthetic-terrain path for Region B. In order: calls
`real_polar_dem.raster_footprint_latlon()` to get the real PGDA tile's own true center and safe
half-width directly from its embedded metadata (Section 8) → calls
`real_polar_dem.load_real_polar_dem()` with those values (not `regions.REGION_MALAPERT`'s
hand-typed ones) → runs the identical crater detection, terrain, illumination, density, and
scoring steps `run_pipeline.py` uses for Tycho → builds the same 2×3 figure grid → prints a
summary that includes the offset (in km) between the real tile's true center and the value
`REGION_MALAPERT` had assumed, computed using the Moon's real radius (`MOON_RADIUS_M` from
`data.py`) rather than Earth's, which was an early self-caught bug (see "bugs found and fixed").
This script is what first surfaced, and then confirmed the fix for, both the window-clamping
bug and the ~33 km location discrepancy described in Sections 3 and 8.

### `scripts/run_real_malapert_extension.py` — TOPSIS + sensitivity, real data
Mirrors `run_extension_analysis.py`'s Region B logic exactly (same `analyze_region` function
signature and outputs), but its `_prepare_real_malapert()` sources data from
`real_polar_dem.load_real_polar_dem()`/`raster_footprint_latlon()` instead of
`synthetic.build_synthetic_dem()`. This is the script behind the real numbers quoted in Section
10 above (17.6 km max shift, 0.0 km TOPSIS/weighted-sum agreement) and in the report's Table
4.4 — the first time this project's TOPSIS/sensitivity extension analysis had been run against
real Malapert elevation data rather than the synthetic prototype.

---

## 13. `notebooks/build_notebook.py` and the notebook itself

**Job:** `build_notebook.py` is a small script that uses the `nbformat` library to
*programmatically* construct the Jupyter notebook cell-by-cell (rather than being hand-typed in
Jupyter), interleaving markdown explanation cells with code cells that mirror the same logic as
`run_pipeline.py`/`run_real_malapert.py`. The resulting notebook
(`landing_site_walkthrough.ipynb`) was then actually re-executed top to bottom
(`jupyter nbconvert --execute`) after the Region B cells were rewritten to load the real PGDA
DEM, and produced zero errors across all 20 cells — so every number and figure in it is a live,
reproducible result, not a static copy-paste from the script. Its Region B section now matches
the report and Sections 3/8 of this document: it loads the real DEM via
`real_polar_dem.raster_footprint_latlon()`/`load_real_polar_dem()`, prints the same ~33 km
offset-discovery numbers described above, and reproduces the real headline results (0 catalog
craters in this small a window, 3 CV detections, best site 85.96°S/2.49°E at 99.9/100) — plus a
short methodology note pointing at the superseded `synthetic.py` prototype for anyone curious
about earlier development history. If asked "why build the notebook with a script instead of
just writing it directly in Jupyter," the honest answer is that it made it easy to keep the
notebook's narration and code perfectly consistent and easy to regenerate — this Region B rewrite
is a concrete example of exactly that regeneration happening in practice.

---

## 14. `yolo_stretch/` — the optional deep-learning experiment

This is clearly separated from the main pipeline (it lives in its own folder and is described
in the report as a "supplementary result," not the primary method) because it uses a
completely different approach — a trainable neural network instead of classical, training-free
computer vision — and its results, honestly, are not good. Being able to explain *why* they're
not good, without being defensive about it, is one of the strongest things you can do in your
evaluation.

### `build_dataset.py` (75 lines)
Converts the source dataset's raw CSV bounding-box annotations into the specific text format
YOLO expects (one line per box: class, center-x, center-y, width, height, all as fractions of
image size between 0 and 1). The one genuinely important detail (lines 39, and the module
docstring): **it reads each image's real pixel dimensions individually** with
`Image.open(img_path).size`, rather than assuming every image is the same size — because the
training images in this dataset are actually 800×800 pixels while the validation images are
1000×1000. An earlier version of this script assumed a single fixed size for everything, which
silently produced out-of-bounds bounding-box coordinates for one split; reading each image's
real size fixed it. This is a good concrete "a bug I found and fixed" story if asked.

### `train.py` (28 lines)
Fine-tunes a pretrained YOLOv8n model (a small, fast object-detection network, originally
trained on everyday photos in the COCO dataset) on the crater dataset for 6 epochs, on CPU
only, at reduced image size (640 instead of the dataset's native ~800–1000) — all three of
those choices (6 epochs, CPU, reduced size) were made specifically to fit inside this
environment's time constraints, and are explicitly labeled as such in a code comment on line
19.

### `RESULTS.md`
Reports the actual metrics honestly: precision 0.018, recall 0.023, mAP50 0.0021 — genuinely
poor numbers — together with three specific, concrete reasons why, each with a stated fix:
too little training data (12 images vs. a typical few hundred to thousands), too few epochs (6
vs. a typical 50–300), and no GPU (CPU-only training is dramatically slower per epoch). **The
loss curves do decrease across training** (Figure 4.3 in the report) — proof the fine-tuning
mechanism itself works correctly end-to-end on real data — it just wasn't given enough data or
training time to become a usable detector. This is presented as a proof-of-concept for the
fine-tuning *workflow*, not a working detector, and the classical-CV baseline in `craters.py`
(90% recall, no training required) remains the pipeline's real, validated method.

---

## 15. Bugs found and fixed along the way (good "tell me about a challenge" material)

1. **Polar region distortion.** First attempt used a fixed degree-box for Malapert, which
   rendered as a thin, squashed sliver rather than a square region — root cause: near the
   pole, a degree of longitude is a much smaller real distance than a degree of latitude.
   Fixed by adding `half_width_km` and the `is_local_km` flag, and reworking `synthetic.py`
   and `viz.py` to build and plot directly in real kilometers for this region.
2. **Crater-detector noise at small radii.** `min_radius_km=3.0` produced 221 detections, most
   pinned right at the minimum size — a sign of noise, not real craters, since 3 km craters are
   barely a couple of pixels wide at this resolution. Raised the floor to 5 km, which removed
   most of the noise-driven detections while keeping recall unchanged (90%).
3. **YOLO label conversion assumed one fixed image size.** Caused an "out of bounds" error
   during training on one split. Fixed by reading each image's true size individually rather
   than assuming a single size across the whole dataset.
4. **Long-running training kept hitting time limits.** The original training plan (8 epochs at
   full image resolution) was too slow for the CPU-only, time-limited environment available
   while building this project. Scaled back to 6 epochs at a smaller image size, and ran
   training in the background while other parts of the project (the notebook, the README) were
   written in parallel.
5. **Real-DEM window-clamping bug (`real_polar_dem.py`).** After downloading the real PGDA
   polar DEM tile for Malapert, the first run against it plotted catalog craters wildly outside
   the visible terrain and matched 0 of 6 nearby catalog craters. Root cause: the requested crop
   box (120×120 km) was larger than the real tile actually covers (~21×21 km), so the pixel
   window had to be clamped to the tile's real edges before reading — but `lat_bounds`/
   `lon_bounds` were being computed from the original, *unclamped* requested corners rather than
   the corners of the window that was *actually* read. Every downstream lat/lon (catalog
   filtering, detection positions, the best-site star, axis labels) was therefore computed in a
   coordinate frame that didn't match the pixels on screen. Fixed by computing bounds from the
   actual clamped pixel window (via `ds.transform` on the clamped row/column indices) and adding
   a `was_clamped` warning print so future runs immediately flag when this situation happens
   again.
6. **Real tile centered ~33 km from where the code assumed (crater vs. massif mixup).** Even
   after fixing the clamping bug, the real tile's own footprint (from
   `raster_footprint_latlon()`) turned out to be centered about 33 km from `regions.py`'s
   `REGION_MALAPERT` coordinate. Investigation (checking the PGDA site list and Malapert
   crater's published catalog entry) showed the original coordinate had actually been set to
   *Malapert crater's* catalogued center, not *Malapert Massif's* — two related but distinct
   named features. Fixed by having `run_real_malapert.py` and
   `run_real_malapert_extension.py` read the real tile's own true center directly from its
   embedded metadata instead of trusting a hand-typed coordinate at all, which makes this class
   of mixup structurally impossible going forward. (An earlier version of the diagnostic print
   for this also used Earth's ~111 km/degree instead of the Moon's ~30.3 km/degree, understating
   the true offset — caught and fixed in the same pass.)

Being able to describe even one of these clearly — what went wrong, how you noticed, and how
you fixed it — is usually worth more in an evaluation than reciting the final numbers.

---

## 16. The most likely hard questions, and how to answer them honestly

**"Did you write all this code yourself?"** Be honest and direct about how you worked, per
your own project's academic-integrity guidelines: say plainly that you used AI assistance to
build the pipeline and draft the report, and that your work has been to understand it deeply
enough to explain and defend every part of it — which is exactly what this document and your
own rewriting of the report are for.

**"Why should I trust the Tycho results and the Malapert ones equally?"** Both now use a real
DEM: Tycho from the global ±60° elevation image, Malapert from NASA GSFC's real 5 m/pixel PGDA
LOLA tile for the site (`real_polar_dem.py`). Both also use the real, independent Head et al.
catalog for crater validation. The one place the two regions still genuinely differ is the
illumination model: Tycho uses the full cosine-incidence law (`mid_latitude_incidence`),
appropriate at its moderate sun-elevation angle; Malapert, where the sun sits almost exactly on
the horizon, uses an explicit height-based proxy (`polar_illumination_proxy`) rather than full
horizon ray-tracing — and that is true regardless of whether the terrain under it is real or
synthetic, since it's a property of the illumination *model*, not the elevation data. That
proxy is stated as such everywhere it's used, not discovered after the fact.

**"Wait — didn't you originally use synthetic terrain for Malapert? What happened?"** Yes, and
it's worth being able to tell this honestly: an early prototype (`synthetic.py`) filled the gap
before the real polar DEM tile had been downloaded, built from real crater positions and a real
published massif height but generated terrain in between. Once the real NASA GSFC PGDA 5
m/pixel tile for the site was downloaded, it was integrated (`real_polar_dem.py`), and every
current report figure and number for Malapert comes from that real data — the synthetic
prototype contributes to none of the final results and is kept in the codebase only as a
historical record of how the project developed. Integrating it also surfaced and fixed two real
bugs (a coordinate-clamping bug, and a ~33 km mixup between Malapert crater's catalogued center
and Malapert Massif's true location) — see "Bugs found and fixed."

**"What would you do differently with more time?"** Replace the polar illumination proxy with
real horizon ray-tracing against the now-available high-resolution DEM; run the YOLOv8
experiment on the full dataset with a GPU and far more epochs; add more criteria (Earth
line-of-sight, geological unit) following Peña-Asensio et al. (2024); and replace the
one-at-a-time sensitivity sweep with full weight-simplex sampling, which can catch effects from
two weights changing together that OAT can miss.

**"What's the single most important scientific result here?"** The 90% crater-detection recall
against a real, independent, published catalog at Tycho — because it's the one result in the
whole project that required no synthetic data and no unvalidated proxy to produce.

**"You added a TOPSIS comparison and a sensitivity analysis — why, and what did they show?"**
Because a weighted-sum formula with hand-picked weights is the most subjective part of the
whole pipeline, and I wanted evidence that it wasn't fragile rather than just asserting it.
TOPSIS — a completely different combination method — picked the exact same best site as the
weighted sum at both regions, and the one-at-a-time sensitivity sweep showed the best site
barely moves under a wide range of plausible reweighting, only shifting at extreme values. Both
are covered in `sensitivity.py` (Section 10 above) and Sections 4.4/5.3 of the report.
