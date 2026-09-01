"""
Named case-study regions for the landing-site suitability pipeline.

REGION_TYCHO is the primary demonstration region: it sits inside the real
DEM's +/-60 deg latitude coverage, so every stage of the pipeline (crater
detection, slope/roughness, illumination, composite score) runs on genuine
per-pixel elevation data and can be checked against a real published crater
catalog.

REGION_MALAPERT is the aerospace-motivated case study: Malapert Massif is
one of NASA's 13 real candidate landing regions for Artemis III. It sits at
~85 deg S, outside this pipeline's DEM coverage, so its analysis in this
project uses (a) the real global crater catalog, which *does* cover the
pole, and (b) a physically-motivated but unvalidated illumination proxy
model rather than a raster-derived one. See illumination.py and the README
"Known limitations" section for exactly what this does and does not prove,
and for the drop-in path to the real 5 m/pixel PGDA DEM.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Region:
    name: str
    lat_center: float
    lon_center: float
    description: str
    has_raster_dem: bool
    half_width_deg: float | None = None   # used for real-DEM regions (degree-based crop)
    half_width_km: float | None = None    # used for synthetic near-pole regions (local flat-km box)


REGION_TYCHO = Region(
    name="Tycho crater region",
    lat_center=-43.3,
    lon_center=-11.2,
    half_width_deg=5.0,
    description=(
        "Tycho crater (43.3S, 11.2W), a young ~85 km rayed highland crater "
        "with a prominent central peak, surrounded by numerous smaller "
        "superimposed craters. Chosen as the primary demo region because "
        "it lies well inside the real DEM's +/-60 deg coverage and gives "
        "the crater detector a rich, real, and independently-cataloged "
        "scene to work with."
    ),
    has_raster_dem=True,
)

REGION_MALAPERT = Region(
    name="Malapert Massif (Artemis III candidate)",
    lat_center=-84.9,
    lon_center=0.4,
    half_width_km=60.0,  # a 120x120 km local box, NOT a degree box -- see note below
    description=(
        "Malapert Massif, a ~5 km high mountain on the rim of Malapert "
        "crater (~85S, near 0 deg longitude), one of NASA's 13 publicly "
        "announced candidate landing regions for Artemis III. Its elevation "
        "and near-polar latitude are exactly why it is a candidate: tall "
        "terrain near the pole can stay sunlit (for power and thermal "
        "management) while also keeping line-of-sight to Earth for "
        "communications. Analyzed here using the global crater catalog "
        "plus a topography-driven illumination *proxy*, not a raster DEM "
        "(see module docstrings for why).\n\n"
        "Note on half_width_km vs half_width_deg: a fixed degree-box "
        "(e.g. +/-2 deg lon) covers wildly different real distances in "
        "latitude vs. longitude this close to the pole -- at 85S, 4 deg of "
        "longitude is only ~11 km of real ground, while 4 deg of latitude "
        "is ~121 km. A degree-box there is a thin sliver, not a square "
        "region. So this region is instead defined as a fixed square in "
        "real kilometers, built directly in a local flat approximation "
        "(valid at this small scale) rather than in lat/lon degrees."
    ),
    has_raster_dem=False,
)
