"""
Module: utils/validators.py
Purpose: Input validation utilities for the ESP Generator pipeline.

Validates file paths, coordinate bounds, and geometry inputs before
processing begins — fail fast, fail clearly.

Standards: PEP 8, PEP 484 (type hints)
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Custom Exception
# ---------------------------------------------------------------------------

class ValidationError(ValueError):
    """Raised when an input fails validation checks.

    Inherits from ValueError so callers can catch it specifically or broadly.
    """
    pass


# ---------------------------------------------------------------------------
# File Validators
# ---------------------------------------------------------------------------

def validate_file_exists(path: Path | str, label: str = "file") -> Path:
    """Ensure a file exists and is not a directory.

    Args:
        path: Path to the file.
        label: Human-readable label used in error messages (e.g. "KML file").

    Returns:
        Resolved :class:`pathlib.Path` object.

    Raises:
        ValidationError: If path does not exist or is a directory.
    """
    resolved = Path(path).resolve()
    if not resolved.exists():
        raise ValidationError(
            f"{label} not found: '{resolved}'. "
            "Please check the file path."
        )
    if resolved.is_dir():
        raise ValidationError(
            f"Expected a file for {label}, but found a directory: '{resolved}'."
        )
    logger.debug("File validated: %s", resolved)
    return resolved


def validate_file_extension(
    path: Path | str,
    allowed_extensions: Sequence[str],
    label: str = "file",
) -> Path:
    """Ensure a file has one of the allowed extensions (case-insensitive).

    Args:
        path: Path to the file.
        allowed_extensions: Sequence of allowed extensions, e.g. ``['.kml', '.kmz']``.
        label: Human-readable label used in error messages.

    Returns:
        Resolved :class:`pathlib.Path` object.

    Raises:
        ValidationError: If the extension is not in the allowed list.
    """
    resolved = validate_file_exists(path, label)
    ext = resolved.suffix.lower()
    normalised = [e.lower() for e in allowed_extensions]
    if ext not in normalised:
        raise ValidationError(
            f"Invalid extension '{ext}' for {label} '{resolved.name}'. "
            f"Allowed: {normalised}."
        )
    logger.debug("Extension '%s' valid for %s.", ext, label)
    return resolved


def validate_kml_or_kmz(path: Path | str) -> Path:
    """Validate that a path points to a readable .kml or .kmz file.

    Args:
        path: Path to the KML or KMZ file.

    Returns:
        Resolved :class:`pathlib.Path` object.

    Raises:
        ValidationError: If the file does not exist or has wrong extension.
    """
    return validate_file_extension(path, [".kml", ".kmz"], label="KML/KMZ alignment file")


def validate_geotiff(path: Path | str) -> Path:
    """Validate that a path points to a readable GeoTIFF DEM file.

    Args:
        path: Path to the GeoTIFF file.

    Returns:
        Resolved :class:`pathlib.Path` object.

    Raises:
        ValidationError: If the file does not exist or has wrong extension.
    """
    return validate_file_extension(
        path, [".tif", ".tiff", ".geotiff", ".asc"], label="DEM Raster file"
    )


# ---------------------------------------------------------------------------
# Coordinate Validators
# ---------------------------------------------------------------------------

def validate_latitude(lat: float, label: str = "latitude") -> float:
    """Validate a WGS84 latitude value.

    Args:
        lat: Latitude in decimal degrees.
        label: Human-readable label for error messages.

    Returns:
        The validated latitude value.

    Raises:
        ValidationError: If latitude is outside [-90, 90].
    """
    if not (-90.0 <= lat <= 90.0):
        raise ValidationError(
            f"Invalid {label} = {lat}. Must be in range [-90, 90] degrees."
        )
    return lat


def validate_longitude(lon: float, label: str = "longitude") -> float:
    """Validate a WGS84 longitude value.

    Args:
        lon: Longitude in decimal degrees.
        label: Human-readable label for error messages.

    Returns:
        The validated longitude value.

    Raises:
        ValidationError: If longitude is outside [-180, 180].
    """
    if not (-180.0 <= lon <= 180.0):
        raise ValidationError(
            f"Invalid {label} = {lon}. Must be in range [-180, 180] degrees."
        )
    return lon


def validate_coordinate_pair(
    lon: float,
    lat: float,
    index: int | None = None,
) -> tuple[float, float]:
    """Validate a single WGS84 (longitude, latitude) coordinate pair.

    Args:
        lon: Longitude in decimal degrees.
        lat: Latitude in decimal degrees.
        index: Optional point index for error context.

    Returns:
        Tuple of (longitude, latitude).

    Raises:
        ValidationError: If either value is out of valid WGS84 range.
    """
    context = f"at index {index}" if index is not None else ""
    validate_longitude(lon, label=f"longitude {context}")
    validate_latitude(lat, label=f"latitude {context}")
    return lon, lat


def validate_coordinate_list(
    coords: list[tuple[float, float, float]],
    min_points: int = 2,
) -> list[tuple[float, float, float]]:
    """Validate a list of (lon, lat, alt) coordinate tuples.

    Checks:
    - Minimum number of points
    - All coordinates within WGS84 bounds
    - No NaN or Inf values
    - No exact duplicate consecutive points (degenerate segments)

    Args:
        coords: List of ``(longitude, latitude, altitude)`` tuples.
        min_points: Minimum required number of coordinate points.

    Returns:
        The validated coordinate list.

    Raises:
        ValidationError: If any check fails.
    """
    if len(coords) < min_points:
        raise ValidationError(
            f"Alignment has only {len(coords)} point(s). "
            f"Minimum required: {min_points}. "
            "Please check the KML file contains a valid LineString."
        )

    duplicate_count = 0
    for i, (lon, lat, alt) in enumerate(coords):
        # NaN / Inf check
        if not all(math.isfinite(v) for v in (lon, lat, alt)):
            raise ValidationError(
                f"Non-finite coordinate at index {i}: "
                f"lon={lon}, lat={lat}, alt={alt}."
            )
        # WGS84 bounds
        validate_coordinate_pair(lon, lat, index=i)

        # Duplicate consecutive points
        if i > 0:
            prev_lon, prev_lat, _ = coords[i - 1]
            if math.isclose(lon, prev_lon, abs_tol=1e-9) and math.isclose(
                lat, prev_lat, abs_tol=1e-9
            ):
                logger.warning(
                    "Duplicate consecutive coordinate at index %d: (%.8f, %.8f). "
                    "Will be ignored during chainage generation.",
                    i,
                    lon,
                    lat,
                )
                duplicate_count += 1

    if duplicate_count > 0:
        logger.warning(
            "%d duplicate consecutive point(s) detected in alignment.", duplicate_count
        )

    logger.info(
        "Coordinate list validated: %d points, %d duplicate(s).",
        len(coords),
        duplicate_count,
    )
    return coords


# ---------------------------------------------------------------------------
# Config Validators
# ---------------------------------------------------------------------------

def validate_config_key(
    config: dict,
    *keys: str,
    label: str = "config",
) -> None:
    """Assert that nested keys exist in a configuration dictionary.

    Args:
        config: The configuration dictionary (loaded from config.yaml).
        *keys: Sequence of keys representing the path, e.g.
            ``'alignment'``, ``'chainage_interval'``.
        label: Human-readable label for error messages.

    Raises:
        ValidationError: If any key in the path is missing.
    """
    node: dict = config
    path_str = ""
    for key in keys:
        path_str = f"{path_str}.{key}" if path_str else key
        if not isinstance(node, dict) or key not in node:
            raise ValidationError(
                f"Missing required config key '{path_str}' in {label}. "
                "Please check config/config.yaml."
            )
        node = node[key]
