"""
Module: modules/read_kml.py
Purpose: Module 1 — Read KML/KMZ alignment file and convert to UTM.

Engineering Basis:
    Railway alignments are captured by DGPS/RTK in WGS84 (EPSG:4326) and
    exported as KML/KMZ polylines. This module:
      1. Detects file type (.kml vs .kmz)
      2. Parses the <LineString> element from KML XML
      3. Validates all coordinates against WGS84 bounds
      4. Auto-detects the UTM zone from the alignment centroid
      5. Reprojects WGS84 → UTM using pyproj (vectorised)
      6. Returns a GeoDataFrame with UTM coordinates + metadata

UTM Zone Formula:
    zone = floor((mean_longitude + 180) / 6) + 1
    India: Zones 42N–47N → EPSG:32642–32647

Output DataFrame Columns:
    point_id   : int    — sequential index (0-based)
    latitude   : float  — WGS84 latitude (degrees)
    longitude  : float  — WGS84 longitude (degrees)
    altitude_m : float  — altitude from KML (0.0 if absent)
    easting    : float  — UTM easting (meters)
    northing   : float  — UTM northing (meters)
    chainage   : float  — cumulative chainage (0.0 — filled by Module 2)
    geometry   : Point  — Shapely Point in UTM CRS

Standards: RDSO / Indian Railways BG
Author   : Railway ESP Generator
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root directory to sys.path for direct module execution
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import shutil
import tempfile
import zipfile
from dataclasses import dataclass, field
from typing import Any
import math

import geopandas as gpd
import numpy as np
import pandas as pd
from lxml import etree
from shapely.geometry import LineString, Point
from shapely.ops import linemerge

from utils.geo_utils import UTMInfo, detect_utm_zone, total_alignment_length, wgs84_to_utm
from utils.logger import get_logger
from utils.validators import validate_coordinate_list, validate_kml_or_kmz

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# KML XML Namespace constants
# Google Earth uses two different namespaces across versions.
# ---------------------------------------------------------------------------
_KML_NAMESPACES: tuple[str, ...] = (
    "http://www.opengis.net/kml/2.2",         # OGC standard KML 2.2
    "http://earth.google.com/kml/2.2",         # Google Earth legacy
    "http://earth.google.com/kml/2.1",         # Older Google Earth
    "http://www.opengis.net/kml/2.3",         # KML 2.3 (future-proof)
)


# ---------------------------------------------------------------------------
# Custom Exception
# ---------------------------------------------------------------------------

class KMLReadError(Exception):
    """Raised when the KML/KMZ file cannot be read or parsed.

    Common causes:
    - File not found (caught earlier by validator)
    - KML contains no LineString elements
    - Malformed XML
    - Coordinate values out of WGS84 bounds
    """
    pass


# ---------------------------------------------------------------------------
# Result Dataclass
# ---------------------------------------------------------------------------

@dataclass
class AlignmentResult:
    """Result returned by :class:`KMLReader`.

    Contains the reprojected alignment points as a GeoDataFrame together
    with engineering metadata. This object is the input to Module 2
    (ChainageGenerator).

    Attributes:
        gdf: GeoDataFrame with columns:
            ``point_id``, ``latitude``, ``longitude``, ``altitude_m``,
            ``easting``, ``northing``, ``chainage``, ``geometry``.
            CRS is set to the detected UTM zone.
        alignment_line: Shapely LineString of the alignment in UTM coordinates.
            Used by Module 2 for interpolating chainage points.
        utm_info: :class:`UTMInfo` named tuple with EPSG code and zone details.
        metadata: Engineering metadata dictionary including:
            - ``source_file``: Absolute path to the input KML/KMZ file.
            - ``total_vertices``: Number of raw KML vertices.
            - ``raw_length_m``: Total alignment length (vertex-to-vertex) in meters.
            - ``mean_latitude``: Centroid latitude used for zone detection.
            - ``mean_longitude``: Centroid longitude used for zone detection.
    """
    gdf: gpd.GeoDataFrame
    alignment_line: LineString
    utm_info: UTMInfo
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"AlignmentResult("
            f"vertices={self.metadata.get('total_vertices', '?')}, "
            f"length={self.metadata.get('raw_length_m', 0.0):.1f} m, "
            f"crs={self.utm_info.epsg_str})"
        )


# ---------------------------------------------------------------------------
# KMLReader Class
# ---------------------------------------------------------------------------

class KMLReader:
    """Read a KML or KMZ railway alignment file and convert to UTM.

    Implements Module 1 of the ESP Generator pipeline. Follows the
    Single Responsibility Principle: this class only reads and reprojects
    the raw alignment. Chainage generation is Module 2.

    Args:
        config: Configuration dictionary loaded from ``config/config.yaml``.

    Example:
        >>> import yaml
        >>> config = yaml.safe_load(open('config/config.yaml'))
        >>> result = KMLReader(config).read('data/alignment.kml')
        >>> print(result.gdf.head())
        >>> print(f"Total length: {result.metadata['raw_length_m']:.1f} m")
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self._alignment_cfg: dict[str, Any] = config.get("alignment", {})
        self._min_points: int = self._alignment_cfg.get("min_points", 2)
        self._reverse_alignment: bool = bool(self._alignment_cfg.get("reverse_alignment", False))
        logger.info("KMLReader initialised. reverse_alignment=%s", self._reverse_alignment)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read(self, kml_path: str | Path) -> AlignmentResult:
        """Read a KML or KMZ file and return a UTM-projected AlignmentResult.

        Processing steps:
          1. Validate file path and extension.
          2. If KMZ: extract to a temporary directory, locate the .kml inside.
          3. Parse KML XML → extract <LineString> coordinates (lon, lat, alt).
          4. Validate all coordinates.
          5. Detect UTM zone from alignment centroid.
          6. Reproject WGS84 → UTM (vectorised).
          7. Build GeoDataFrame + Shapely LineString.
          8. Return AlignmentResult.

        Args:
            kml_path: Path to the ``.kml`` or ``.kmz`` file.

        Returns:
            :class:`AlignmentResult` with GeoDataFrame, LineString, and metadata.

        Raises:
            KMLReadError: If the file cannot be parsed or contains no alignment.
        """
        path = validate_kml_or_kmz(kml_path)
        logger.info("Reading alignment from: %s", path)

        # Step 1 — Handle KMZ (unzip)
        kml_file, tmp_dir = self._resolve_kml(path)

        try:
            # Step 2 — Parse KML XML
            raw_coords, raw_segments = self._parse_kml(kml_file)
            logger.info("Parsed %d raw coordinate points from KML.", len(raw_coords))

            # Step 3 — Validate coordinates
            validate_coordinate_list(raw_coords, min_points=self._min_points)

            # Step 4 — Remove exact duplicate consecutive points
            raw_coords = self._deduplicate(raw_coords)

            # Step 4b — Strip Civil 3D CAD marker symbols (origin circles/arrows)
            raw_coords = self._strip_cad_markers(raw_coords)

            # Step 5 — Compute centroid for UTM zone detection
            lons = [c[0] for c in raw_coords]
            lats = [c[1] for c in raw_coords]
            alts = [c[2] for c in raw_coords]
            mean_lon = float(np.mean(lons))
            mean_lat = float(np.mean(lats))

            # Step 6 — Detect UTM zone
            utm_info = detect_utm_zone(mean_lon, mean_lat)

            # Step 7 — Reproject WGS84 → UTM (vectorised)
            eastings, northings = wgs84_to_utm(lons, lats, utm_info)

            # Step 8 — Compute true 3D WGS84 geodesic ellipsoid alignment length
            raw_length_m = total_alignment_length(eastings, northings, lons=lons, lats=lats, alts=alts)
            logger.info(
                "Raw alignment length (vertex-to-vertex): %.2f m (%.3f km)",
                raw_length_m,
                raw_length_m / 1000.0,
            )

            # Step 9 — Build GeoDataFrame
            gdf = self._build_geodataframe(
                lons=lons,
                lats=lats,
                alts=alts,
                eastings=eastings,
                northings=northings,
                utm_info=utm_info,
            )

            # Step 10 — Build Shapely LineString (UTM coordinates)
            utm_pairs = list(zip(eastings.tolist(), northings.tolist()))
            if self._reverse_alignment:
                logger.info(
                    "reverse_alignment=true: flipping %d UTM vertices so Chainage 0 is at the END of the KML.",
                    len(utm_pairs),
                )
                utm_pairs = utm_pairs[::-1]
                # Also reverse the coordinate lists used for the GeoDataFrame
                lons = lons[::-1]
                lats = lats[::-1]
                alts = alts[::-1]
                eastings = eastings[::-1]
                northings = northings[::-1]
            alignment_line = LineString(utm_pairs)
            
            # Step 11 — Extract segment lengths for placemark alignment elements
            segment_lengths: list[float] = []
            if len(raw_segments) > 1:
                for seg in raw_segments:
                    if len(seg) > 1:
                        s_lons = [c[0] for c in seg]
                        s_lats = [c[1] for c in seg]
                        s_alts = [c[2] for c in seg]
                        s_e, s_n = wgs84_to_utm(s_lons, s_lats, utm_info)
                        s_len = total_alignment_length(s_e, s_n, lons=s_lons, lats=s_lats, alts=s_alts)
                        segment_lengths.append(s_len)

            metadata: dict[str, Any] = {
                "source_file": str(path.resolve()),
                "total_vertices": len(raw_coords),
                "raw_length_m": raw_length_m,
                "raw_length_km": raw_length_m / 1000.0,
                "mean_latitude": mean_lat,
                "mean_longitude": mean_lon,
                "source_crs": "EPSG:4326",
                "target_crs": utm_info.epsg_str,
                "utm_zone": utm_info.zone,
                "utm_hemisphere": utm_info.hemisphere,
                "crs_name": utm_info.crs_name,
                "segment_lengths": segment_lengths,
            }

            result = AlignmentResult(
                gdf=gdf,
                alignment_line=alignment_line,
                utm_info=utm_info,
                metadata=metadata,
            )

            logger.info("AlignmentResult built: %r", result)
            return result

        finally:
            # Always clean up the temp directory if we created one
            if tmp_dir is not None:
                shutil.rmtree(tmp_dir, ignore_errors=True)
                logger.debug("Temporary directory cleaned up.")

    # ------------------------------------------------------------------
    # Private Methods
    # ------------------------------------------------------------------

    def _resolve_kml(
        self,
        path: Path,
    ) -> tuple[Path, str | None]:
        """Resolve the actual .kml file, extracting from KMZ if necessary.

        Args:
            path: Validated file path (.kml or .kmz).

        Returns:
            Tuple of (kml_file_path, tmp_dir_path_or_None).
            If the input was a KMZ, tmp_dir is the temporary directory
            that must be cleaned up by the caller.

        Raises:
            KMLReadError: If the KMZ contains no .kml file.
        """
        if path.suffix.lower() == ".kmz":
            logger.info("Detected KMZ archive — extracting...")
            tmp_dir = tempfile.mkdtemp(prefix="esp_kml_")
            try:
                with zipfile.ZipFile(path, "r") as z:
                    z.extractall(tmp_dir)
            except zipfile.BadZipFile as exc:
                shutil.rmtree(tmp_dir, ignore_errors=True)
                raise KMLReadError(
                    f"KMZ file is corrupted or not a valid ZIP: '{path}'. "
                    f"Error: {exc}"
                ) from exc

            kml_files = sorted(Path(tmp_dir).rglob("*.kml"))
            if not kml_files:
                shutil.rmtree(tmp_dir, ignore_errors=True)
                raise KMLReadError(
                    f"No .kml file found inside KMZ archive: '{path}'. "
                    "Verify the KMZ was exported correctly from Google Earth."
                )

            # Prefer 'doc.kml' (Google Earth default) or the first found
            kml_file = next(
                (f for f in kml_files if f.name.lower() == "doc.kml"),
                kml_files[0],
            )
            logger.info("KML extracted: %s", kml_file)
            return kml_file, tmp_dir

        # Already a .kml file
        return path, None

    def _parse_kml(
        self,
        kml_path: Path,
    ) -> tuple[list[tuple[float, float, float]], list[list[tuple[float, float, float]]]]:
        """Parse a KML file and extract LineString coordinates.

        Tries all known KML namespaces. Handles both:
        - ``<LineString><coordinates>`` (standard)
        - ``<MultiGeometry>`` containing nested LineStrings (flattened)

        The KML coordinate format is: lon,lat[,alt] separated by whitespace.

        Args:
            kml_path: Path to the resolved .kml file.

        Returns:
            List of ``(longitude, latitude, altitude)`` tuples.

        Raises:
            KMLReadError: If no LineString coordinates are found.
        """
        try:
            tree = etree.parse(str(kml_path))
        except etree.XMLSyntaxError as exc:
            raise KMLReadError(
                f"KML file has invalid XML syntax: '{kml_path}'. "
                f"Error: {exc}"
            ) from exc

        root = tree.getroot()
        all_segments: list[list[tuple[float, float, float]]] = []

        # ── Civil 3D / Tessellated Chord fast-path ────────────────────────────
        chord_containers = root.xpath(
            ".//*[local-name()='Placemark' or (local-name()='Folder' and not(.//*[local-name()='Placemark']))]"
            "[count(.//*[local-name()='LineString']) > 5]"
        )
        if not chord_containers:
            chord_containers = root.xpath(
                ".//*[local-name()='Folder' or local-name()='Placemark']"
                "[count(.//*[local-name()='LineString']) > 5]"
            )

        fast_path_coords: list[tuple[float, float, float]] | None = None
        if chord_containers:
            def _container_score(c):
                tot_len = 0.0
                for el in c.xpath(".//*[local-name()='coordinates']"):
                    pts = self._parse_coord_text(el.text or "")
                    if len(pts) >= 2:
                        for i in range(len(pts) - 1):
                            dx = (pts[i + 1][0] - pts[i][0]) * 103000.0
                            dy = (pts[i + 1][1] - pts[i][1]) * 111000.0
                            seg_len = math.hypot(dx, dy)
                            if seg_len > 0.5:
                                tot_len += seg_len
                return tot_len

            best_container = max(chord_containers, key=_container_score)
            raw_segs: list[list[tuple[float, float, float]]] = []
            for coords_el in best_container.xpath(".//*[local-name()='coordinates']"):
                seg = self._parse_coord_text(coords_el.text or "")
                if len(seg) >= 2:
                    raw_segs.append(seg)

            chained_tokens = self._chain_segments(raw_segs)
            if len(chained_tokens) >= 2:
                fast_path_coords = chained_tokens

        aecc_placemarks = root.xpath(".//*[local-name()='Placemark'][contains(translate(.//*[local-name()='name']/text(), 'abcdefghijklmnopqrstuvwxyz', 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 'ALIGNMENT') or contains(translate(.//*[local-name()='name']/text(), 'abcdefghijklmnopqrstuvwxyz', 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 'AECC')]")
        
        target_elements = []
        target_name = (self.config or {}).get("kml_reader", {}).get("target_alignment_name") or (self.config or {}).get("target_alignment_name")

        if aecc_placemarks:
            logger.info("Detected %d Civil 3D AECC_ALIGNMENT placemark(s).", len(aecc_placemarks))
            matched_pms = []
            if target_name:
                for pm in aecc_placemarks:
                    full_text = "".join(pm.xpath(".//text()"))
                    if target_name.upper() in full_text.upper():
                        matched_pms.append(pm)
                if matched_pms:
                    logger.info("Found %d placemark(s) matching target alignment name '%s'.", len(matched_pms), target_name)
            
            if not matched_pms:
                for pm in aecc_placemarks:
                    desc_text = "".join(pm.xpath(".//*[local-name()='description']//text()"))
                    if "0+000" in desc_text or "0.000" in desc_text or "General Overall Length" in desc_text:
                        matched_pms.append(pm)

            selected_pms = matched_pms if matched_pms else aecc_placemarks
            logger.info("Selected %d primary Civil 3D alignment placemark(s).", len(selected_pms))
            for pm in selected_pms:
                coords_elems = pm.xpath(".//*[local-name()='coordinates']")
                target_elements.extend(coords_elems)
        
        if not target_elements:
            target_elements = root.xpath(".//*[local-name()='coordinates']")

        if target_elements:
            logger.debug("Parsing %d <coordinates> element(s).", len(target_elements))
            for elem in target_elements:
                segment = self._parse_coord_text(elem.text or "")
                if segment and len(segment) > 1:
                    lons = [c[0] for c in segment]
                    lats = [c[1] for c in segment]
                    mean_lon = float(np.mean(lons))
                    mean_lat = float(np.mean(lats))
                    
                    if not (68.0 <= mean_lon <= 97.0 and 6.0 <= mean_lat <= 38.0):
                        continue
                        
                    extent_m = math.hypot((max(lons) - min(lons)) * 111000.0, (max(lats) - min(lats)) * 111000.0)
                    if 0.1 <= extent_m <= 50000.0:
                        all_segments.append(segment)

        if not all_segments and not fast_path_coords:
            raise KMLReadError(
                f"No <LineString><coordinates> element found in KML: '{kml_path}'. "
                "Ensure the KML was exported with a LineString/Path geometry, "
                "not as Points or Polygons."
            )

        # 1. Folder-level linemerge to avoid multi-layer intersection splits
        folder_lines: list[LineString] = []
        for folder_el in root.xpath(".//*[local-name()='Folder']"):
            f_segs: list[LineString] = []
            for coords_el in folder_el.xpath(".//*[local-name()='coordinates']"):
                pts = self._parse_coord_text(coords_el.text or "")
                if len(pts) >= 2:
                    f_segs.append(LineString(pts))
            if f_segs:
                try:
                    m_f = linemerge(f_segs)
                    if m_f.geom_type == "LineString":
                        folder_lines.append(m_f)
                    elif m_f.geom_type == "MultiLineString":
                        folder_lines.extend(list(m_f.geoms))
                except Exception:
                    folder_lines.extend(f_segs)

        lines = folder_lines if folder_lines else [LineString(seg) for seg in all_segments if len(seg) > 1]
        coords: list[tuple[float, float, float]] = []

        if lines:
            try:
                merged = linemerge(lines)
                if merged.geom_type == 'LineString':
                    logger.info("Successfully stitched continuous alignment LineString (%.2fm).", merged.length * 111000.0)
                    coords = list(merged.coords)
                elif merged.geom_type == 'MultiLineString':
                    geoms = sorted(list(merged.geoms), key=lambda x: x.length, reverse=True)
                    main_coords = list(geoms.pop(0).coords)
                    
                    while geoms:
                        best_idx = -1
                        best_d = float('inf')
                        conn_start = False
                        rev = False
                        for i, g in enumerate(geoms):
                            gc = list(g.coords)
                            d1 = (gc[-1][0] - main_coords[0][0]) ** 2 + (gc[-1][1] - main_coords[0][1]) ** 2
                            if d1 < best_d:
                                best_d = d1
                                best_idx = i
                                conn_start = True
                                rev = False
                            d2 = (gc[0][0] - main_coords[0][0]) ** 2 + (gc[0][1] - main_coords[0][1]) ** 2
                            if d2 < best_d:
                                best_d = d2
                                best_idx = i
                                conn_start = True
                                rev = True
                            d3 = (gc[0][0] - main_coords[-1][0]) ** 2 + (gc[0][1] - main_coords[-1][1]) ** 2
                            if d3 < best_d:
                                best_d = d3
                                best_idx = i
                                conn_start = False
                                rev = False
                            d4 = (gc[-1][0] - main_coords[-1][0]) ** 2 + (gc[-1][1] - main_coords[-1][1]) ** 2
                            if d4 < best_d:
                                best_d = d4
                                best_idx = i
                                conn_start = False
                                rev = True

                        if best_d < (1e-5) ** 2:  # Exact endpoint match (~0.1m gap tolerance)
                            bg = geoms.pop(best_idx)
                            add_c = list(bg.coords)
                            if rev:
                                add_c.reverse()
                            if conn_start:
                                main_coords = add_c[:-1] + main_coords
                            else:
                                main_coords = main_coords + add_c[1:]
                        else:
                            break

                    logger.info("Chained connected alignment pieces into %.2fm mainline (%d vertices).", LineString(main_coords).length * 111000.0, len(main_coords))
                    coords = main_coords
                else:
                    coords = all_segments[0]
            except Exception as exc:
                logger.warning("Linemerge failed (%s); selecting longest single segment.", exc)
                segment_lengths = [LineString(seg).length for seg in all_segments]
                longest_idx = int(np.argmax(segment_lengths))
                coords = all_segments[longest_idx]
        elif fast_path_coords:
            coords = fast_path_coords
        else:
            coords = all_segments[0]

        if fast_path_coords and len(coords) > 1:
            len_merged = LineString(coords).length * 111000.0
            len_fast = LineString(fast_path_coords).length * 111000.0
            if len_fast > len_merged * 1.5:
                coords = fast_path_coords

        logger.debug("Total coordinates extracted from the Main Line KML segment: %d", len(coords))
        return coords, all_segments

    @staticmethod
    def _parse_coord_text(
        text: str,
    ) -> list[tuple[float, float, float]]:
        """Parse KML coordinate text into (lon, lat, alt) tuples.

        KML coordinate format:
            ``lon,lat,alt lon,lat,alt ...`` (whitespace-separated tokens)
        or: ``lon,lat lon,lat ...``          (without altitude)

        Args:
            text: Raw text content of the <coordinates> element.

        Returns:
            List of ``(longitude, latitude, altitude)`` float tuples.

        Raises:
            KMLReadError: If a token cannot be parsed as numeric values.
        """
        import re
        cleaned_text = re.sub(r'\s*,\s*', ',', text.strip())
        coords: list[tuple[float, float, float]] = []
        for token in cleaned_text.split():
            token = token.strip()
            if not token:
                continue
            parts = token.split(",")
            if len(parts) < 2 or not parts[0] or not parts[1]:
                logger.warning("Skipping malformed coordinate token: '%s'", token)
                continue
            try:
                lon = float(parts[0])
                lat = float(parts[1])
                alt = float(parts[2]) if len(parts) > 2 and parts[2].strip() else 0.0
            except ValueError as exc:
                raise KMLReadError(
                    f"Cannot parse coordinate token '{token}': {exc}. "
                    "Check the KML file for non-numeric coordinate values."
                ) from exc
            coords.append((lon, lat, alt))
        return coords

    @staticmethod
    def _deduplicate(
        coords: list[tuple[float, float, float]],
    ) -> list[tuple[float, float, float]]:
        """Remove exact duplicate consecutive coordinate points.

        Duplicate consecutive points (same lon AND lat to 9 decimal places)
        would create zero-length segments, causing division-by-zero in
        downstream chainage and geometry calculations.

        Args:
            coords: Input coordinate list.

        Returns:
            Cleaned coordinate list with duplicates removed.
        """
        if not coords:
            return coords

        cleaned: list[tuple[float, float, float]] = [coords[0]]
        removed = 0

        for lon, lat, alt in coords[1:]:
            prev_lon, prev_lat, _ = cleaned[-1]
            if abs(lon - prev_lon) < 1e-9 and abs(lat - prev_lat) < 1e-9:
                removed += 1
                continue
            cleaned.append((lon, lat, alt))

        if removed:
            logger.info(
                "Removed %d duplicate consecutive coordinate point(s). "
                "%d points remain.",
                removed,
                len(cleaned),
            )
        return cleaned

    @staticmethod
    def _strip_cad_markers(
        coords: list[tuple[float, float, float]],
        marker_seg_threshold_m: float = 2.0,
        gap_threshold_m: float = 20.0,
        min_marker_points: int = 3,
    ) -> list[tuple[float, float, float]]:
        """Collapse Civil 3D origin/end marker symbols into single centroids.

        Civil 3D KML exports often include small graphical symbols (origin
        marker circles, alignment start arrows) as coordinate vertices.
        These tiny clusters of points (diameter < 1m) add spurious path
        length from the circle perimeter and distort downstream geometry.

        Instead of deleting the marker entirely (which would lose the
        alignment origin and the tangent approach to the first track
        vertex), this method **collapses** each marker cluster into a
        single centroid point that represents the true alignment origin.

        Detection criteria:
            - A cluster of >= min_marker_points consecutive vertices with
              ALL inter-point distances < marker_seg_threshold_m.
            - Followed (or preceded) by a gap > gap_threshold_m to the
              next/previous actual alignment vertex.

        Applied at both the leading and trailing ends of the alignment.

        Args:
            coords: Input coordinate list (lon, lat, alt) after deduplication.
            marker_seg_threshold_m: Maximum segment length within a marker
                cluster (default 2.0m). Civil 3D markers typically have
                inter-point distances of 0.01–0.1m.
            gap_threshold_m: Minimum gap length after/before the marker
                cluster to confirm it is a symbol (default 20.0m).
            min_marker_points: Minimum number of tightly-spaced points
                to constitute a marker symbol (default 3).

        Returns:
            Cleaned coordinate list with marker clusters collapsed to centroids.
        """
        if len(coords) < min_marker_points + 2:
            return coords

        def _approx_dist_m(
            c1: tuple[float, float, float],
            c2: tuple[float, float, float],
        ) -> float:
            """Approximate distance between two WGS84 (lon, lat, alt) tuples in meters."""
            dlat_m = (c2[1] - c1[1]) * 111_000.0
            mean_lat_rad = math.radians((c1[1] + c2[1]) / 2.0)
            dlon_m = (c2[0] - c1[0]) * 111_000.0 * math.cos(mean_lat_rad)
            return math.hypot(dlat_m, dlon_m)

        def _centroid(
            pts: list[tuple[float, float, float]],
        ) -> tuple[float, float, float]:
            """Compute centroid of a list of (lon, lat, alt) tuples."""
            n = len(pts)
            return (
                sum(p[0] for p in pts) / n,
                sum(p[1] for p in pts) / n,
                sum(p[2] for p in pts) / n,
            )

        # ── Detect leading marker ────────────────────────────────────────
        leading_end = 0  # index of first non-marker vertex
        for i in range(1, len(coords) - 1):
            seg_len = _approx_dist_m(coords[i - 1], coords[i])
            if seg_len < marker_seg_threshold_m:
                # Still inside a tightly-spaced cluster
                continue
            else:
                # First large segment encountered
                if i >= min_marker_points and seg_len > gap_threshold_m:
                    leading_end = i
                    logger.info(
                        "Detected Civil 3D origin marker: %d vertices collapsed "
                        "to centroid (inter-point < %.1fm, gap to alignment: %.1fm).",
                        i, marker_seg_threshold_m, seg_len,
                    )
                break

        # ── Detect trailing marker ───────────────────────────────────────
        trailing_start = len(coords)  # index of first trailing marker vertex
        for i in range(len(coords) - 2, leading_end, -1):
            seg_len = _approx_dist_m(coords[i], coords[i + 1])
            if seg_len < marker_seg_threshold_m:
                continue
            else:
                trailing_count = len(coords) - 1 - i
                if trailing_count >= min_marker_points and seg_len > gap_threshold_m:
                    trailing_start = i + 1
                    logger.info(
                        "Detected Civil 3D end marker: %d vertices collapsed "
                        "to centroid (inter-point < %.1fm, gap from alignment: %.1fm).",
                        trailing_count, marker_seg_threshold_m, seg_len,
                    )
                break

        if leading_end > 0 or trailing_start < len(coords):
            # Build cleaned list: [leading centroid] + middle + [trailing centroid]
            result: list[tuple[float, float, float]] = []

            if leading_end > 0:
                centroid_lead = _centroid(coords[:leading_end])
                result.append(centroid_lead)
                total_collapsed = leading_end - 1  # collapsed N points into 1
            else:
                total_collapsed = 0

            result.extend(coords[leading_end:trailing_start])

            if trailing_start < len(coords):
                centroid_trail = _centroid(coords[trailing_start:])
                result.append(centroid_trail)
                total_collapsed += len(coords) - trailing_start - 1

            logger.info(
                "Collapsed %d CAD marker vertices into centroid(s). "
                "%d alignment points remain.",
                total_collapsed, len(result),
            )
            return result

        return coords

    @staticmethod
    def _build_geodataframe(
        lons: list[float],
        lats: list[float],
        alts: list[float],
        eastings: np.ndarray,
        northings: np.ndarray,
        utm_info: UTMInfo,
    ) -> gpd.GeoDataFrame:
        """Construct the output GeoDataFrame from processed coordinates.

        Args:
            lons: List of WGS84 longitudes.
            lats: List of WGS84 latitudes.
            alts: List of altitudes (from KML, 0.0 if absent).
            eastings: UTM easting array (meters).
            northings: UTM northing array (meters).
            utm_info: UTM zone information for CRS assignment.

        Returns:
            :class:`geopandas.GeoDataFrame` with all alignment attributes
            and CRS set to the detected UTM zone.
        """
        n = len(lons)
        geometry = [Point(e, n_) for e, n_ in zip(eastings.tolist(), northings.tolist())]

        gdf = gpd.GeoDataFrame(
            data={
                "point_id": list(range(n)),
                "latitude": lats,
                "longitude": lons,
                "altitude_m": alts,
                "easting": eastings.tolist(),
                "northing": northings.tolist(),
                "chainage": [0.0] * n,   # Filled by Module 2 (ChainageGenerator)
            },
            geometry=geometry,
            crs=utm_info.epsg_str,
        )

        logger.debug(
            "GeoDataFrame built: %d rows, CRS=%s.", len(gdf), utm_info.epsg_str
        )
        return gdf

    @staticmethod
    def _chain_segments(
        segments: list[list[tuple[float, float, float]]],
    ) -> list[tuple[float, float, float]]:
        """Spatially chain unordered KML chord segments into a single ordered polyline."""
        if not segments:
            return []

        from collections import defaultdict

        def snap(pt: tuple[float, float, float]) -> tuple[float, float, float]:
            return (round(pt[0], 7), round(pt[1], 7), round(pt[2], 7) if len(pt) > 2 else 0.0)

        snapped_segs = [[snap(p) for p in seg] for seg in segments if len(seg) >= 2]
        if not snapped_segs:
            return []

        endpoint_map: dict[tuple[float, float, float], list[int]] = defaultdict(list)
        for idx, seg in enumerate(snapped_segs):
            endpoint_map[seg[0]].append(idx)
            endpoint_map[seg[-1]].append(idx)

        # Start from segment index 0 — document order reflects engineer's digitizing direction.
        # The old West/South heuristic was wrong for East→West or North→South alignments.
        # If reverse_alignment=true in config, the caller will flip the whole chain afterward.
        start_idx = 0

        visited = {start_idx}
        chain = list(snapped_segs[start_idx])

        # Orient the starting segment so its HEAD connects forward, not backward
        if len(endpoint_map[chain[-1]]) > 1 and len(endpoint_map[chain[0]]) == 1:
            pass  # chain[0] is a dead-end → it IS the start; keep orientation
        elif len(endpoint_map[chain[0]]) > 1 and len(endpoint_map[chain[-1]]) == 1:
            chain.reverse()  # chain[-1] is a dead-end → flip so dead-end becomes start

        while len(visited) < len(snapped_segs):
            tail = chain[-1]
            candidates = [idx for idx in endpoint_map[tail] if idx not in visited]

            if not candidates:
                best_idx = -1
                best_dist = float('inf')
                for idx in range(len(snapped_segs)):
                    if idx in visited:
                        continue
                    seg = snapped_segs[idx]
                    d_head = (seg[0][0] - tail[0])**2 + (seg[0][1] - tail[1])**2
                    d_tail = (seg[-1][0] - tail[0])**2 + (seg[-1][1] - tail[1])**2
                    if d_head < best_dist:
                        best_dist = d_head
                        best_idx = idx
                    if d_tail < best_dist:
                        best_dist = d_tail
                        best_idx = idx

                if best_idx != -1 and best_dist < (0.0005)**2:  # ~50m threshold
                    candidates = [best_idx]
                else:
                    break

            next_idx = candidates[0]
            visited.add(next_idx)
            next_seg = list(snapped_segs[next_idx])

            d_tail = (next_seg[-1][0] - tail[0])**2 + (next_seg[-1][1] - tail[1])**2
            d_head = (next_seg[0][0] - tail[0])**2 + (next_seg[0][1] - tail[1])**2
            if d_tail < d_head:
                next_seg.reverse()

            chain.extend(next_seg[1:])

        return chain


if __name__ == '__main__':
    import yaml

    config = {}
    config_file = Path("config/config.yaml")
    if config_file.exists():
        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

    reader = KMLReader(config)
    sample_kml = Path("data/sample_alignment.kml")
    if sample_kml.exists():
        result = reader.read(sample_kml)
        print(f"\n✅ Read complete! Length: {result.alignment_line.length:.2f}m | CRS: {result.utm_info.epsg_str}")
    else:
        print("KMLReader initialised. Call reader.read('<path_to_kml_or_kmz>') to process an alignment.")
