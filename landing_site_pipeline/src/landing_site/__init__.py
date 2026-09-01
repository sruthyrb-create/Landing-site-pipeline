"""
landing_site
============

A small research-style pipeline that scores lunar terrain for landing-site
suitability by combining three signals:

1. Crater hazard   - detected via classical computer vision on real DEM data,
                      cross-checked against a published crater catalog.
2. Slope/roughness  - derived directly from real elevation data.
3. Illumination     - modeled with two different approaches depending on
                      latitude (see `illumination.py`).

See the project README for data sources and methodology notes.
"""

from . import data, terrain, craters, illumination, scoring, viz, regions, synthetic, sensitivity  # noqa: F401

try:
    
    from . import real_polar_dem  # noqa: F401
except ImportError:
    pass
