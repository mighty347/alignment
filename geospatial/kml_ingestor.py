"""
Module: future_alignment_studio/geospatial/kml_ingestor.py
Purpose: Ingest and standardize railway alignment centerlines from survey KML/KMZ files.

Standards:
- IRPWM / RDSO Metric Chainage Pegging (default 20.0m intervals).
- Automatic UTM Zone determination to convert geodetic coordinates to metric Cartesian space.
- Zero hardcoded parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Tuple
from xml.etree import ElementTree as ET
import zipfile

import yaml
from pyproj import Transformer
from shapely.geometry import LineString, Point

try:
    from alignment.utils.logger import get_logger
except ImportError:
    try:
        from utils.logger import get_logger
    except ImportError:
        from future_alignment_studio.utils.logger import get_logger

logger = get_logger("KMLIngestor")
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config/alignment_config.yaml"


@dataclass
class AlignmentStationPoint:
    """Represents a single pegged station along the alignment centerline."""
    station_index: int
    chainage_m: float
    chainage_display: str       # e.g., "0+020.00"
    easting_m: float            # Projected UTM Easting (X) in meters
    northing_m: float           # Projected UTM Northing (Y) in meters
    lon_deg: float              # Original WGS-84 Longitude
    lat_deg: float              # Original WGS-84 Latitude


@lru_cache(maxsize=4)
def _read_ingestor_yaml(config_path: Path, mtime_ns: int) -> dict[str, Any]:
    """Read railway standards from YAML; keyed by mtime so edits reload automatically."""
    del mtime_ns
    with config_path.open(encoding="utf-8") as handle:
        return dict(yaml.safe_load(handle) or {})


def _format_chainage_display(chainage_m: float) -> str:
    """Format chainage (m) into Indian Railways 'KM+MMM.MM' notation."""
    sign = "-" if chainage_m < 0.0 else ""
    abs_m = abs(chainage_m)
    km = int(abs_m // 1000.0)
    rem = abs_m % 1000.0
    return f"{sign}{km}+{rem:06.2f}"


def _parse_coord_text(raw_text: str) -> List[Tuple[float, float]]:
    """Parse KML whitespace-delimited `lon,lat[,alt]` tokens into deduplicated (lon, lat) pairs."""
    pts: List[Tuple[float, float]] = []
    for chunk in raw_text.strip().split():
        parts = chunk.split(",")
        if len(parts) < 2:
            continue
        try:
            lon = float(parts[0])
            lat = float(parts[1])
        except ValueError:
            continue
        if not (math.isfinite(lon) and math.isfinite(lat)):
            continue
        if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
            continue
        if not pts or pts[-1] != (lon, lat):
            pts.append((lon, lat))
    return pts


class KMLAlignmentIngestor:
    """
    Parses and standardizes railway alignment centerlines from survey KML/KMZ files.
    """

    def __init__(
        self,
        chainage_interval_m: Optional[float] = None,
        *,
        config: Optional[Mapping[str, Any]] = None,
        config_path: Optional[Path] = None,
    ) -> None:
        raw_cfg: dict[str, Any] = {}
        resolved = (config_path or _DEFAULT_CONFIG_PATH).resolve()
        if resolved.exists():
            raw_cfg = _read_ingestor_yaml(resolved, resolved.stat().st_mtime_ns)

        railway_std: dict[str, Any] = dict(raw_cfg.get("railway_standards", {}))
        if config is not None and isinstance(config.get("railway_standards"), Mapping):
            railway_std.update(config["railway_standards"])

        interval = float(
            railway_std["chainage_interval_m"] if chainage_interval_m is None else chainage_interval_m
        )
        if not math.isfinite(interval) or interval <= 0.0:
            raise ValueError(f"chainage_interval_m must be finite and positive; got {interval!r}.")

        self.chainage_interval_m = interval
        self.chainage_precision = int(railway_std["station_chainage_precision"])
        self.coord_precision = int(railway_std["projected_coord_precision"])
        self.geo_precision = int(railway_std["geodetic_coord_precision"])
        self.terminal_station_min_gap_m = float(railway_std["terminal_station_min_gap_m"])
        logger.info(
            "Initialized KMLAlignmentIngestor with chainage interval: %.1fm",
            self.chainage_interval_m,
        )

    def extract_raw_coordinates(
        self,
        file_path: str | Path,
        line_index: Optional[int] = None,
    ) -> List[Tuple[float, float]]:
        """
        Extracts raw (lon, lat) coordinate tuples from a .kml or .kmz file.
        Returns coordinates in order from start to end of the selected corridor.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Alignment survey file not found: {path}")

        kml_content: bytes
        if path.suffix.lower() == ".kmz":
            logger.info("Decompressing KMZ archive: %s", path.name)
            with zipfile.ZipFile(path, "r") as z:
                kml_entries = [name for name in z.namelist() if name.lower().endswith(".kml")]
                if not kml_entries:
                    raise ValueError(f"No .kml document found inside {path.name}")
                kml_filename = next(
                    (name for name in kml_entries if name.lower().endswith("doc.kml")),
                    kml_entries[0],
                )
                kml_content = z.read(kml_filename)
        else:
            kml_content = path.read_bytes()

        return self.parse_kml_xml(kml_content, line_index=line_index)

    def parse_kml_corridors(self, xml_bytes: bytes) -> List[Tuple[str, List[Tuple[float, float]]]]:
        """Extract named LineString corridors from KML XML without mixing Point markers."""
        if not xml_bytes or not xml_bytes.strip():
            raise ValueError("KML XML content is empty.")
        root = ET.fromstring(xml_bytes)

        corridors: List[Tuple[str, List[Tuple[float, float]]]] = []
        stack: List[Tuple[ET.Element, str]] = [(root, "")]
        while stack:
            node, parent_name = stack.pop()
            tag = node.tag.rsplit("}", 1)[-1]
            name = parent_name
            if tag in ("Placemark", "Folder", "Document"):
                name = next(
                    (
                        (child.text or "").strip()
                        for child in node
                        if child.tag.rsplit("}", 1)[-1] == "name" and (child.text or "").strip()
                    ),
                    parent_name,
                )
            if tag == "LineString":
                for child in node:
                    if child.tag.rsplit("}", 1)[-1] == "coordinates" and child.text:
                        pts = _parse_coord_text(child.text)
                        if len(pts) >= 2:
                            corridors.append((name or f"Line {len(corridors) + 1}", pts))
            stack.extend((child, name) for child in reversed(list(node)))

        if not corridors:
            for idx, node in enumerate(root.iter()):
                if node.tag.rsplit("}", 1)[-1] == "coordinates" and node.text:
                    pts = _parse_coord_text(node.text)
                    if len(pts) >= 2:
                        corridors.append((f"Line {idx + 1}", pts))

        if not corridors:
            raise ValueError("No valid LineString corridors found in the provided KML file.")
        return corridors

    def parse_kml_xml(
        self,
        xml_bytes: bytes,
        line_index: Optional[int] = None,
    ) -> List[Tuple[float, float]]:
        """
        Parses KML XML structure and extracts coordinates for the selected LineString corridor.
        """
        corridors = self.parse_kml_corridors(xml_bytes)
        idx = 0 if line_index is None else line_index
        if idx < 0 or idx >= len(corridors):
            raise ValueError(
                f"line_index {idx} out of range for KML containing {len(corridors)} corridor(s)."
            )
        selected_name, selected_pts = corridors[idx]
        logger.info(
            "Successfully extracted %d raw survey points from KML corridor '%s'",
            len(selected_pts),
            selected_name,
        )
        return selected_pts

    @staticmethod
    def detect_utm_epsg(lon: float, lat: float) -> int:
        """
        Determines the optimal UTM Zone EPSG code based on longitude and latitude.
        Zone number = floor((lon + 180) / 6) + 1 clamped to [1, 60].
        Northern hemisphere: EPSG: 32600 + zone
        Southern hemisphere: EPSG: 32700 + zone
        """
        if not (math.isfinite(lon) and math.isfinite(lat)):
            raise ValueError(f"Coordinates must be finite; got ({lon!r}, {lat!r}).")
        if not (-180.0 <= lon <= 180.0 and -80.0 <= lat <= 84.0):
            raise ValueError(
                f"Coordinate ({lon}, {lat}) is outside valid UTM coverage (lon [-180, 180], lat [-80, 84])."
            )
        zone = max(1, min(60, int(math.floor((lon + 180.0) / 6.0)) + 1))
        base = 32600 if lat >= 0.0 else 32700
        epsg = base + zone
        logger.info(
            "Auto-detected UTM EPSG:%d (Zone %d %s) for coordinate (%.4f, %.4f)",
            epsg, zone, "N" if lat >= 0 else "S", lon, lat,
        )
        return epsg

    def _build_station(
        self,
        station_idx: int,
        chainage_m: float,
        east_m: float,
        north_m: float,
        rev_transformer: Transformer,
    ) -> AlignmentStationPoint:
        """Construct a rounded AlignmentStationPoint at the given UTM position and chainage."""
        lon_deg, lat_deg = rev_transformer.transform(east_m, north_m)
        return AlignmentStationPoint(
            station_index=station_idx,
            chainage_m=round(chainage_m, self.chainage_precision),
            chainage_display=_format_chainage_display(chainage_m),
            easting_m=round(east_m, self.coord_precision),
            northing_m=round(north_m, self.coord_precision),
            lon_deg=round(lon_deg, self.geo_precision),
            lat_deg=round(lat_deg, self.geo_precision),
        )

    def process_and_resample(
        self,
        raw_coords: Sequence[Tuple[float, float]],
        target_epsg: Optional[int] = None,
        start_chainage_m: float = 0.0,
    ) -> Tuple[List[AlignmentStationPoint], int]:
        """
        Projects coordinates to metric UTM space and resamples the path
        at exact chainage intervals (e.g. every 20m) starting from `start_chainage_m`.
        """
        if len(raw_coords) < 2:
            raise ValueError("At least 2 points required to create an alignment line.")
        if not math.isfinite(start_chainage_m):
            raise ValueError(f"start_chainage_m must be finite; got {start_chainage_m!r}.")

        for idx, (lon, lat) in enumerate(raw_coords):
            if not (math.isfinite(lon) and math.isfinite(lat)):
                raise ValueError(f"Coordinate {idx} is non-finite: ({lon}, {lat}).")
            if not (-180.0 <= lon <= 180.0 and -80.0 <= lat <= 84.0):
                raise ValueError(f"Coordinate {idx} ({lon}, {lat}) is outside UTM bounds.")

        if target_epsg is None:
            avg_lon = sum(pt[0] for pt in raw_coords) / len(raw_coords)
            avg_lat = sum(pt[1] for pt in raw_coords) / len(raw_coords)
            target_epsg = self.detect_utm_epsg(avg_lon, avg_lat)

        fwd_transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{target_epsg}", always_xy=True)
        rev_transformer = Transformer.from_crs(f"EPSG:{target_epsg}", "EPSG:4326", always_xy=True)

        projected_pts: List[Tuple[float, float]] = []
        for lon, lat in raw_coords:
            east, north = fwd_transformer.transform(lon, lat)
            if not projected_pts or (east, north) != projected_pts[-1]:
                projected_pts.append((east, north))

        if len(projected_pts) < 2:
            raise ValueError("Projected alignment contains fewer than 2 distinct points.")

        metric_line = LineString(projected_pts)
        total_length_m = metric_line.length
        if total_length_m <= 0.0:
            raise ValueError("Alignment centerline has zero length.")

        logger.info(
            "Total projected alignment length: %.2f meters (%.3f km)",
            total_length_m, total_length_m / 1000.0,
        )

        stations: List[AlignmentStationPoint] = []
        rel_ch = 0.0
        station_idx = 0
        while rel_ch <= total_length_m + 1e-9:
            pt: Point = metric_line.interpolate(min(rel_ch, total_length_m))
            stations.append(
                self._build_station(station_idx, start_chainage_m + rel_ch, pt.x, pt.y, rev_transformer)
            )
            rel_ch += self.chainage_interval_m
            station_idx += 1

        last_rel_ch = (len(stations) - 1) * self.chainage_interval_m
        if (total_length_m - last_rel_ch) > self.terminal_station_min_gap_m:
            end_pt: Point = metric_line.interpolate(total_length_m)
            terminal = self._build_station(
                station_idx, start_chainage_m + total_length_m, end_pt.x, end_pt.y, rev_transformer
            )
            prev = stations[-1]
            if (
                terminal.chainage_m > prev.chainage_m
                and (terminal.easting_m, terminal.northing_m) != (prev.easting_m, prev.northing_m)
            ):
                stations.append(terminal)

        logger.info("Generated %d pegged stations along the centerline", len(stations))
        return stations, target_epsg

