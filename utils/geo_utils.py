"""
Module: utils/geo_utils.py
Purpose: Shared geospatial helper functions for the ESP Generator.

Provides UTM zone detection, EPSG code building, and coordinate
transformation utilities used by all geospatial modules.

Engineering Basis:
    UTM Zone formula:
        zone = floor((longitude + 180) / 6) + 1
    EPSG codes:
        Northern hemisphere: 32600 + zone  (e.g. Zone 43N → EPSG:32643)
        Southern hemisphere: 32700 + zone

Standards: PEP 8, PEP 484, RDSO / Indian Railways BG
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import NamedTuple

# Auto-configure GDAL_DATA and PROJ_LIB for Windows conda envs
if "GDAL_DATA" not in os.environ:
    gdal_cand = Path(sys.prefix) / "Library" / "share" / "gdal"
    if gdal_cand.exists():
        os.environ["GDAL_DATA"] = str(gdal_cand)

if "PROJ_LIB" not in os.environ:
    proj_cand = Path(sys.prefix) / "Library" / "share" / "proj"
    if proj_cand.exists():
        os.environ["PROJ_LIB"] = str(proj_cand)
        os.environ["PROJ_DATA"] = str(proj_cand)

import numpy as np
from pyproj import CRS, Transformer

from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Named Tuples / Data Containers
# ---------------------------------------------------------------------------

class UTMInfo(NamedTuple):
    """Information about a detected UTM coordinate system.

    Attributes:
        zone: UTM zone number (1–60).
        hemisphere: 'N' (north) or 'S' (south).
        epsg_code: EPSG numeric code for the UTM zone.
        epsg_str: EPSG string, e.g. ``'EPSG:32643'``.
        crs_name: Human-readable CRS name from pyproj.
    """
    zone: int
    hemisphere: str
    epsg_code: int
    epsg_str: str
    crs_name: str


# ---------------------------------------------------------------------------
# UTM Zone Detection
# ---------------------------------------------------------------------------

def detect_utm_zone(mean_longitude: float, mean_latitude: float) -> UTMInfo:
    """Detect the UTM zone for an alignment centroid.

    Uses the standard UTM zone formula. For alignments crossing a zone
    boundary, the centroid longitude determines the single zone used for
    the entire project — this is acceptable for alignments < 300 km.

    Formula:
        zone = floor((mean_longitude + 180) / 6) + 1
        EPSG = 32600 + zone   (Northern hemisphere)
        EPSG = 32700 + zone   (Southern hemisphere)

    Indian Railway Zone Coverage:
        Longitude 66°E–98°E → Zones 42–47
        EPSG:32642 to EPSG:32647

    Args:
        mean_longitude: Mean longitude of the alignment in decimal degrees.
        mean_latitude: Mean latitude of the alignment in decimal degrees.

    Returns:
        :class:`UTMInfo` named tuple with full CRS details.

    Example:
        >>> info = detect_utm_zone(77.2, 28.6)   # Delhi
        >>> info.epsg_str
        'EPSG:32643'
    """
    zone: int = math.floor((mean_longitude + 180.0) / 6.0) + 1
    hemisphere: str = "N" if mean_latitude >= 0.0 else "S"
    epsg_code: int = 32600 + zone if hemisphere == "N" else 32700 + zone
    epsg_str: str = f"EPSG:{epsg_code}"

    # Verify with pyproj
    crs = CRS.from_epsg(epsg_code)
    crs_name: str = crs.name

    logger.info(
        "UTM zone detected: Zone %d%s | %s | %.4f°E, %.4f°%s",
        zone,
        hemisphere,
        epsg_str,
        mean_longitude,
        abs(mean_latitude),
        hemisphere,
    )

    return UTMInfo(
        zone=zone,
        hemisphere=hemisphere,
        epsg_code=epsg_code,
        epsg_str=epsg_str,
        crs_name=crs_name,
    )


# ---------------------------------------------------------------------------
# Coordinate Transformation
# ---------------------------------------------------------------------------

def build_transformer(
    source_epsg: str = "EPSG:4326",
    target_epsg: str | None = None,
    target_epsg_code: int | None = None,
) -> Transformer:
    """Build a pyproj Transformer from WGS84 to a target CRS.

    Always uses ``always_xy=True`` to ensure the axis order is
    (longitude, latitude) → (easting, northing) regardless of CRS definition.

    Args:
        source_epsg: Source CRS string. Default: ``'EPSG:4326'`` (WGS84).
        target_epsg: Target CRS string, e.g. ``'EPSG:32643'``.
        target_epsg_code: Target EPSG integer (alternative to target_epsg).

    Returns:
        :class:`pyproj.Transformer` instance.

    Raises:
        ValueError: If neither target_epsg nor target_epsg_code is provided.

    Example:
        >>> t = build_transformer(target_epsg_code=32643)
        >>> easting, northing = t.transform(77.2090, 28.6139)
    """
    if target_epsg is None and target_epsg_code is None:
        raise ValueError(
            "Provide either 'target_epsg' (string) or 'target_epsg_code' (int)."
        )
    dst = target_epsg if target_epsg is not None else f"EPSG:{target_epsg_code}"

    transformer = Transformer.from_crs(
        crs_from=source_epsg,
        crs_to=dst,
        always_xy=True,
    )
    logger.debug("Transformer built: %s → %s", source_epsg, dst)
    return transformer


def wgs84_to_utm(
    longitudes: list[float] | np.ndarray,
    latitudes: list[float] | np.ndarray,
    utm_info: UTMInfo,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproject WGS84 (lon, lat) arrays to UTM (easting, northing).

    Uses vectorised pyproj transformation — O(n) time, much faster
    than point-by-point transformation.

    Args:
        longitudes: Array of longitudes in decimal degrees.
        latitudes: Array of latitudes in decimal degrees.
        utm_info: :class:`UTMInfo` with the target UTM CRS.

    Returns:
        Tuple of (eastings, northings) as NumPy float64 arrays, in meters.

    Example:
        >>> lons = [77.2, 77.5, 77.8]
        >>> lats = [28.6, 28.7, 28.8]
        >>> utm_info = detect_utm_zone(77.5, 28.7)
        >>> eastings, northings = wgs84_to_utm(lons, lats, utm_info)
    """
    lons_arr = np.asarray(longitudes, dtype=np.float64)
    lats_arr = np.asarray(latitudes, dtype=np.float64)

    transformer = build_transformer(target_epsg=utm_info.epsg_str)
    eastings, northings = transformer.transform(lons_arr, lats_arr)

    eastings = np.asarray(eastings, dtype=np.float64)
    northings = np.asarray(northings, dtype=np.float64)

    logger.debug(
        "Transformed %d points to %s. "
        "Easting range: [%.1f, %.1f] m, Northing range: [%.1f, %.1f] m.",
        len(lons_arr),
        utm_info.epsg_str,
        eastings.min(),
        eastings.max(),
        northings.min(),
        northings.max(),
    )
    return eastings, northings


# ---------------------------------------------------------------------------
# Segment Distance (Euclidean in UTM)
# ---------------------------------------------------------------------------

def segment_lengths(
    eastings: np.ndarray,
    northings: np.ndarray,
) -> np.ndarray:
    """Compute Euclidean segment lengths between consecutive UTM points.

    For a sequence of n points, returns n-1 distances.

    Formula:
        d_i = sqrt((E_{i+1} - E_i)^2 + (N_{i+1} - N_i)^2)

    Args:
        eastings: 1-D array of easting coordinates (meters).
        northings: 1-D array of northing coordinates (meters).

    Returns:
        1-D NumPy array of length n-1 containing segment distances in meters.

    Raises:
        ValueError: If arrays have different lengths or fewer than 2 points.
    """
    if len(eastings) != len(northings):
        raise ValueError(
            f"eastings length ({len(eastings)}) != northings length ({len(northings)})."
        )
    if len(eastings) < 2:
        raise ValueError(
            f"Need at least 2 points to compute segments. Got {len(eastings)}."
        )
    dE = np.diff(eastings)
    dN = np.diff(northings)
    lengths = np.sqrt(dE**2 + dN**2)
    return lengths


def wgs84_geodesic_length(
    longitudes: np.ndarray | list[float],
    latitudes: np.ndarray | list[float],
    altitudes: np.ndarray | list[float] | None = None,
) -> float:
    """Compute exact WGS84 ellipsoid geodesic 2D/3D length in meters.

    Uses pyproj.Geod(ellps='WGS84') for true geodesic horizontal distance and
    optionally incorporates 3D elevation differences: sqrt(d_horizontal^2 + dz^2).

    Args:
        longitudes: Array of WGS84 longitude coordinates.
        latitudes: Array of WGS84 latitude coordinates.
        altitudes: Optional array of elevation/altitude values in meters.

    Returns:
        Total geodesic length in meters as a float.
    """
    from pyproj import Geod
    geod = Geod(ellps="WGS84")
    lons = np.asarray(longitudes, dtype=float)
    lats = np.asarray(latitudes, dtype=float)
    if len(lons) < 2:
        return 0.0
    _, _, d_horiz = geod.inv(lons[:-1], lats[:-1], lons[1:], lats[1:])
    d_horiz = np.abs(d_horiz)

    if altitudes is not None:
        alts = np.asarray(altitudes, dtype=float)
        if len(alts) == len(lons):
            dz = np.diff(alts)
            d_3d = np.sqrt(d_horiz**2 + dz**2)
            return float(np.sum(d_3d))

    return float(np.sum(d_horiz))


def total_alignment_length(
    eastings: np.ndarray,
    northings: np.ndarray,
    lons: np.ndarray | list[float] | None = None,
    lats: np.ndarray | list[float] | None = None,
    alts: np.ndarray | list[float] | None = None,
) -> float:
    """Compute the total alignment length in meters (sum of all segments).

    If lons/lats are provided, computes true WGS84 ellipsoid geodesic distance (2D/3D);
    otherwise computes flat UTM grid length.

    Args:
        eastings: 1-D array of easting coordinates.
        northings: 1-D array of northing coordinates.
        lons: Optional 1-D array of WGS84 longitude coordinates.
        lats: Optional 1-D array of WGS84 latitude coordinates.
        alts: Optional 1-D array of altitude/elevation coordinates.

    Returns:
        Total length in meters as a float.
    """
    if lons is not None and lats is not None:
        return wgs84_geodesic_length(lons, lats, altitudes=alts)
    return float(segment_lengths(eastings, northings).sum())


# ---------------------------------------------------------------------------
# Format Helpers
# ---------------------------------------------------------------------------

def format_chainage(chainage_m: float, format_pattern: Optional[str] = None) -> str:
    """Format a chainage value in Indian Railways or custom format templates.

    Args:
        chainage_m: Chainage in meters from the origin.
        format_pattern: Optional format template string. Supported templates include:
            - 'CH-{km}+{m:03d}'       -> 'CH-1+250'
            - '{km}+{m:03d}'          -> '1+250'
            - 'Km {km:03d}+{m:03d}'   -> 'Km 001+250'
            - 'Km {km}+{m:03d}'       -> 'Km 1+250'
            - 'KM {km}+{m:03d}'       -> 'KM 1+250'
            - '{ch:.0f}m'             -> '1250m'
            - 'Km {ch_km:.3f}'        -> 'Km 1.250'
            If None, defaults to 'Km {km:03d}+{m:03d}'.

    Returns:
        Formatted chainage string.
    """
    if chainage_m is None or not math.isfinite(chainage_m):
        if format_pattern and "CH-" in format_pattern:
            return "CH-0+000"
        return "Km 000+000"

    sign_str = "-" if chainage_m < 0 else ""
    abs_ch = abs(chainage_m)
    km_abs = int(abs_ch // 1000)
    m_abs = int(round(abs_ch % 1000))
    if m_abs >= 1000:
        km_abs += 1
        m_abs -= 1000

    km_signed = km_abs if chainage_m >= 0 else -km_abs

    if not format_pattern:
        if chainage_m < 0:
            return f"-Km {km_abs:03d}+{m_abs:03d}"
        return f"Km {km_abs:03d}+{m_abs:03d}"

    fmt = format_pattern
    try:
        res = fmt.format(
            sign=sign_str,
            km=km_signed,
            km_abs=km_abs,
            m=m_abs,
            ch=chainage_m,
            ch_abs=abs_ch,
            ch_km=chainage_m / 1000.0,
            ch_km_abs=abs_ch / 1000.0,
        )
        if chainage_m < 0 and "{sign}" not in fmt and not res.startswith("-"):
            if res.startswith("CH-"):
                res = res.replace("CH-", "CH--", 1)
            else:
                res = f"-{res}"
        return res
    except Exception:
        if "CH-" in fmt:
            return f"CH-{sign_str}{km_abs}+{m_abs:03d}"
        if chainage_m < 0:
            return f"-Km {km_abs:03d}+{m_abs:03d}"
        return f"Km {km_abs:03d}+{m_abs:03d}"
