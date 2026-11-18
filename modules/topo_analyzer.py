"""
Module: modules/topo_analyzer.py
Purpose: Module 10 — Topomap GIS Structure & Proposed Bridge Analyzer.

Engineering Basis:
    Reads Topomap GIS survey shapefiles (DRAIN, CULVERT, ROAD, BUILDING) and
    spatially intersects them with the proposed railway alignment. Automatically
    recommends proposed bridges, RCC box culverts, Road Under Bridges (RUB/LHS),
    and flags house/building infringements within the Right-of-Way (ROW).

Standards: RDSO GE:G-1, Indian Railways Bridge Manual (IRBM Para 102)
Author   : Railway ESP Generator
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString

from modules.read_kml import AlignmentResult
from utils.geo_utils import format_chainage
from utils.logger import get_logger

logger = get_logger(__name__)


class TopoAnalyzerError(Exception):
    """Custom exception for Topo Analyzer errors."""
    pass


@dataclass
class ProposedStructure:
    """Dataclass representing a proposed bridge, culvert, or subway.

    Attributes:
        chainage_m: Crossing station chainage in meters.
        structure_type: 'CULVERT', 'MINOR_BRIDGE', 'MAJOR_BRIDGE', 'RUB_LHS', 'ROB'.
        structure_name: Human-readable description (e.g. '2x2.0m RCC Box Culvert').
        span_config: Span specification (e.g. '2x2.0m', '1x5.5m').
        crossing_feature: Feature name (e.g. 'Nalla / Drain', 'Earthen Road').
        clearance_req_m: Required vertical clearance in meters.
        metadata: Engineering dictionary.
    """
    chainage_m: float
    structure_type: str
    structure_name: str
    span_config: str
    crossing_feature: str
    clearance_req_m: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BuildingInfringement:
    """Dataclass representing a house or building inside the ROW boundary."""
    chainage_m: float
    building_type: str
    offset_m: float
    area_sqm: float
    action_required: str


@dataclass
class TopoAnalysisResult:
    """Result object returned by TopoAnalyzer.

    Attributes:
        structures_df: DataFrame of all proposed bridges, culverts, and subways.
        infringements_df: DataFrame of all house/building ROW infringements.
        structures: List of ProposedStructure objects.
        metadata: Engineering metadata.
    """
    structures_df: pd.DataFrame
    infringements_df: pd.DataFrame
    structures: list[ProposedStructure]
    metadata: dict[str, Any] = field(default_factory=dict)


class TopoAnalyzer:
    """Analyzes Topomap GIS shapefiles to propose bridges, culverts, and RUBs.

    Implements Module 10 of the ESP Generator pipeline.

    Args:
        config: Configuration dictionary loaded from config.yaml.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        self.logger.info("Initialized %s", self.__class__.__name__)

        self.row_width_m: float = float(config.get("formation", {}).get("row_width_m", 30.0))
        self.outputs_dir = Path(config.get("paths", {}).get("outputs_dir", "outputs"))
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

    def analyze(
        self,
        align_res: AlignmentResult,
        topo_dir: str | Path | None = None,
    ) -> TopoAnalysisResult:
        """Spatially intersect alignment with topomap GIS shapefiles.

        Args:
            align_res: Output from Module 1 containing alignment geometry.
            topo_dir: Path to Topomap shapefiles directory.

        Returns:
            TopoAnalysisResult containing proposed bridges and infringements.
        """
        self.logger.info("Starting Topomap GIS Structure Analysis...")

        try:
            # Default topo directory search
            if topo_dir is None or not Path(topo_dir).exists():
                search_paths = [
                    Path(r"C:\KanthiVada\Kanthi_Topo.kml"),
                    Path(r"C:\KanthiVada\Kanthi\Kanthi_Topomap"),
                    Path(r"D:\dataset\Kachpura\kachhpura data\kachhpura process data\KACHPURA_TOPOMAP_SHP"),
                    Path(r"D:\ESP\ROB_Topomap_SHP"),
                    Path("data/topomap_shp"),
                ]
                for p in search_paths:
                    if p.exists():
                        topo_dir = p
                        break

            if topo_dir is None or not Path(topo_dir).exists():
                self.logger.warning("No Topomap shapefile directory found. Returning empty bridge schedule.")
                return TopoAnalysisResult(
                    structures_df=pd.DataFrame(),
                    infringements_df=pd.DataFrame(),
                    structures=[],
                    metadata={"status": "No Topo Data"}
                )

            topo_path = Path(topo_dir)
            self.logger.info("Using Topomap SHP directory: %s", topo_path)

            if hasattr(align_res, "alignment_line") and align_res.alignment_line is not None:
                alignment_geom: LineString = align_res.alignment_line
            else:
                alignment_geom = align_res.gdf.geometry.iloc[0]

            if hasattr(align_res, "utm_info") and align_res.utm_info is not None:
                target_crs = align_res.utm_info.epsg_str
            else:
                target_crs = align_res.gdf.crs

            proposed_structures: list[ProposedStructure] = []
            infringements: list[BuildingInfringement] = []

            # ---------------------------------------------------------
            # 1. Drainage & Stream Crossings (Culverts & Bridges)
            # ---------------------------------------------------------
            drain_layers = [
                ("DRAIN.shp", "Nalla / Drain", "CULVERT", "1x2.0m RCC Box Culvert", "1x2.0m", 0.6),
                ("CULVERT.shp", "Existing Culvert", "CULVERT", "2x2.0m RCC Box Culvert", "2x2.0m", 0.6),
                ("RIVER.shp", "River Stream", "MINOR_BRIDGE", "2x6.0m PSC Slab Bridge", "2x6.0m", 1.0),
            ]

            def _find_vector_file(folder: Path, base_name: str) -> Path | None:
                for ext in [".shp", ".dxf", ".geojson", ".json"]:
                    candidate = folder / f"{base_name}{ext}"
                    if candidate.exists():
                        return candidate
                for f in folder.glob("*"):
                    if f.stem.lower() == base_name.lower() and f.suffix.lower() in [".shp", ".dxf", ".geojson", ".json"]:
                        return f
                return None

            for layer_key, feat_name, st_type, st_desc, span, cl_req in drain_layers:
                base_stem = layer_key.split(".")[0]
                vec_file = _find_vector_file(topo_path, base_stem)
                if vec_file is None:
                    continue

                gdf_layer = gpd.read_file(vec_file)
                if gdf_layer.crs != target_crs:
                    gdf_layer = gdf_layer.to_crs(target_crs)

                for _, row in gdf_layer.iterrows():
                    geom = row.geometry
                    if geom is None or geom.is_empty:
                        continue
                    if geom.intersects(alignment_geom):
                        ix = geom.intersection(alignment_geom)
                        pt = ix if ix.geom_type == "Point" else ix.centroid
                        ch_m = float(alignment_geom.project(pt))

                        proposed_structures.append(
                            ProposedStructure(
                                chainage_m=ch_m,
                                structure_type=st_type,
                                structure_name=st_desc,
                                span_config=span,
                                crossing_feature=feat_name,
                                clearance_req_m=cl_req,
                                metadata={"geom_type": geom.geom_type}
                            )
                        )
                        self.logger.info("Detected %s at Chainage %s", st_desc, format_chainage(ch_m))

            # ---------------------------------------------------------
            # 2. Road Crossings (RUB / LHS / Subways)
            # ---------------------------------------------------------
            road_layers = [
                ("EARTHEN-ROAD.shp", "Earthen Road", "RUB_LHS", "1x5.5m Limited Height Subway", "1x5.5m", 4.5),
                ("CARRIAGEWAY-PAVEDSHOULDER.shp", "Paved Road", "RUB_LHS", "1x7.5m Road Under Bridge", "1x7.5m", 5.5),
            ]

            for layer_key, feat_name, st_type, st_desc, span, cl_req in road_layers:
                base_stem = layer_key.split(".")[0]
                vec_file = _find_vector_file(topo_path, base_stem)
                if vec_file is None:
                    continue

                gdf_layer = gpd.read_file(vec_file)
                if gdf_layer.crs != target_crs:
                    gdf_layer = gdf_layer.to_crs(target_crs)

                for _, row in gdf_layer.iterrows():
                    geom = row.geometry
                    if geom is None or geom.is_empty:
                        continue
                    if geom.intersects(alignment_geom):
                        ix = geom.intersection(alignment_geom)
                        pt = ix if ix.geom_type == "Point" else ix.centroid
                        ch_m = float(alignment_geom.project(pt))

                        # Deduplicate: skip if an identical structure exists within 25m
                        is_duplicate = False
                        for prev_s in proposed_structures:
                            if prev_s.structure_type == st_type and abs(ch_m - prev_s.chainage_m) < 25.0:
                                is_duplicate = True
                                break

                        if not is_duplicate:
                            proposed_structures.append(
                                ProposedStructure(
                                    chainage_m=ch_m,
                                    structure_type=st_type,
                                    structure_name=st_desc,
                                    span_config=span,
                                    crossing_feature=feat_name,
                                    clearance_req_m=cl_req,
                                    metadata={"geom_type": geom.geom_type}
                                )
                            )
                            self.logger.info("Detected %s at Chainage %s", st_desc, format_chainage(ch_m))

            # Sort proposed structures by chainage
            proposed_structures.sort(key=lambda s: s.chainage_m)

            # ---------------------------------------------------------
            # 3. House & Building Infringements (within ROW buffer)
            # ---------------------------------------------------------
            row_buffer = alignment_geom.buffer(self.row_width_m / 2.0)
            bldg_layers = [
                ("PUCCA-BUILDING.shp", "Pucca Concrete Building"),
                ("KACHHA-BUILDING.shp", "Kachha House"),
                ("HUT.shp", "Hut / Temporary Structure"),
            ]

            for shp_name, bldg_type in bldg_layers:
                shp_file = topo_path / shp_name
                if not shp_file.exists():
                    continue

                gdf_bldg = gpd.read_file(shp_file)
                if gdf_bldg.crs != target_crs:
                    gdf_bldg = gdf_bldg.to_crs(target_crs)

                for _, row in gdf_bldg.iterrows():
                    geom = row.geometry
                    if geom is None or geom.is_empty:
                        continue
                    if geom.intersects(row_buffer):
                        centroid = geom.centroid
                        ch_m = float(alignment_geom.project(centroid))
                        dist_m = float(alignment_geom.distance(centroid))
                        area_m2 = float(geom.area) if geom.geom_type == "Polygon" else 25.0

                        action = "Land Acquisition & Demolition" if dist_m < 10.0 else "ROW Clearance Notice"
                        infringements.append(
                            BuildingInfringement(
                                chainage_m=ch_m,
                                building_type=bldg_type,
                                offset_m=dist_m,
                                area_sqm=area_m2,
                                action_required=action
                            )
                        )

            # Sort infringements by chainage
            infringements.sort(key=lambda b: b.chainage_m)

            # Convert to DataFrames
            struct_records = [
                {
                    "chainage_m": s.chainage_m,
                    "chainage_formatted": format_chainage(s.chainage_m),
                    "structure_type": s.structure_type,
                    "structure_name": s.structure_name,
                    "span_config": s.span_config,
                    "crossing_feature": s.crossing_feature,
                    "clearance_req_m": s.clearance_req_m,
                }
                for s in proposed_structures
            ]
            df_struct = pd.DataFrame(struct_records)

            infr_records = [
                {
                    "chainage_m": b.chainage_m,
                    "chainage_formatted": format_chainage(b.chainage_m),
                    "building_type": b.building_type,
                    "offset_m": round(b.offset_m, 2),
                    "area_sqm": round(b.area_sqm, 2),
                    "action_required": b.action_required,
                }
                for b in infringements
            ]
            df_infr = pd.DataFrame(infr_records)

            self.logger.info(
                "Topo Analysis complete: %d structures proposed, %d building infringements flagged.",
                len(proposed_structures),
                len(infringements)
            )

            meta = {
                "topo_dir": str(topo_path),
                "total_structures": len(proposed_structures),
                "total_infringements": len(infringements),
            }

            return TopoAnalysisResult(
                structures_df=df_struct,
                infringements_df=df_infr,
                structures=proposed_structures,
                metadata=meta
            )

        except Exception as err:
            self.logger.error("Topo Analysis failed: %s", str(err))
            raise TopoAnalyzerError(f"Topo Analysis failed: {err}") from err

    def export_csv(self, result: TopoAnalysisResult, output_dir: str | Path) -> list[Path]:
        """Export Bridge Schedule and Building Infringements to CSV files.

        Args:
            result: TopoAnalysisResult from analyze().
            output_dir: Directory path to save CSVs.

        Returns:
            List of Path objects for exported CSV files.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        struct_csv = out_path / "proposed_bridge_schedule.csv"
        infr_csv = out_path / "building_infringements_schedule.csv"

        result.structures_df.to_csv(struct_csv, index=False)
        result.infringements_df.to_csv(infr_csv, index=False)

        self.logger.info("Exported Bridge Schedule: %s", struct_csv)
        self.logger.info("Exported Building Infringements: %s", infr_csv)

        return [struct_csv, infr_csv]

    def merge_google_maps_detections(
        self,
        topo_result: TopoAnalysisResult,
        gm_result: Any,
    ) -> TopoAnalysisResult:
        """Merges features detected via Google Maps API into TopoAnalysisResult structures.

        Args:
            topo_result: Existing TopoAnalysisResult from GIS shapefiles.
            gm_result: GoogleMapsDetectionResult from GoogleMapsDetector.

        Returns:
            Updated TopoAnalysisResult containing combined structures.
        """
        if gm_result is None or getattr(gm_result, "features_df", pd.DataFrame()).empty:
            return topo_result

        existing_structs = list(topo_result.structures)
        existing_chs = [s.chainage_m for s in existing_structs]

        detected_features = getattr(gm_result, "detected_features", [])
        added_count = 0

        for feat in detected_features:
            # Avoid duplicate structure proposals within 30m of existing GIS structures
            if any(abs(feat.chainage_m - ex_ch) < 30.0 for ex_ch in existing_chs):
                continue

            new_struct = ProposedStructure(
                chainage_m=feat.chainage_m,
                structure_type=feat.proposed_structure,
                structure_name=feat.structure_name,
                span_config=feat.span_config,
                crossing_feature=feat.feature_name,
                clearance_req_m=feat.clearance_req_m,
                metadata={"source": "Google Maps API", "api_type": feat.raw_api_type}
            )
            existing_structs.append(new_struct)
            existing_chs.append(feat.chainage_m)
            added_count += 1

        # Sort merged structures by chainage
        existing_structs.sort(key=lambda s: s.chainage_m)

        # Re-build DataFrame
        records = []
        for s in existing_structs:
            records.append({
                "chainage_m": s.chainage_m,
                "chainage_str": format_chainage(s.chainage_m),
                "structure_type": s.structure_type,
                "structure_name": s.structure_name,
                "span_config": s.span_config,
                "crossing_feature": s.crossing_feature,
                "clearance_req_m": s.clearance_req_m,
                "source": s.metadata.get("source", "GIS Topomap"),
            })

        merged_df = pd.DataFrame(records)
        self.logger.info("Merged %d Google Maps API detected structure(s) into Bridge Schedule.", added_count)

        return TopoAnalysisResult(
            structures_df=merged_df,
            infringements_df=topo_result.infringements_df,
            structures=existing_structs,
            metadata={**topo_result.metadata, "google_maps_merged": True, "gm_added_count": added_count}
        )

    def merge_osm_detections(
        self,
        topo_result: TopoAnalysisResult,
        osm_features: list[Any],
    ) -> TopoAnalysisResult:
        """Merges features detected from OSM SpatiaLite into TopoAnalysisResult structures."""
        if not osm_features:
            return topo_result

        existing_structs = list(topo_result.structures)
        existing_chs = [s.chainage_m for s in existing_structs]
        added_count = 0

        for feat in osm_features:
            # Avoid duplicate structure proposals within 30m of existing GIS structures
            if any(abs(feat.chainage_m - ex_ch) < 30.0 for ex_ch in existing_chs):
                continue

            new_struct = ProposedStructure(
                chainage_m=feat.chainage_m,
                structure_type=feat.proposed_structure,
                structure_name=feat.feature_name,
                span_config=feat.span_config,
                crossing_feature=feat.feature_name,
                clearance_req_m=feat.clearance_req_m,
                metadata={"source": "OSM SpatiaLite DB", "osm_id": feat.osm_id}
            )
            existing_structs.append(new_struct)
            existing_chs.append(feat.chainage_m)
            added_count += 1

        existing_structs.sort(key=lambda s: s.chainage_m)

        records = []
        for s in existing_structs:
            records.append({
                "chainage_m": s.chainage_m,
                "chainage_str": format_chainage(s.chainage_m),
                "structure_type": s.structure_type,
                "structure_name": s.structure_name,
                "span_config": s.span_config,
                "crossing_feature": s.crossing_feature,
                "clearance_req_m": s.clearance_req_m,
                "source": s.metadata.get("source", "GIS Topomap"),
            })

        merged_df = pd.DataFrame(records)
        self.logger.info("Merged %d OSM SpatiaLite detected structure(s) into Bridge Schedule.", added_count)

        return TopoAnalysisResult(
            structures_df=merged_df,
            infringements_df=topo_result.infringements_df,
            structures=existing_structs,
            metadata={**topo_result.metadata, "osm_spatialite_merged": True, "osm_added_count": added_count}
        )

    def merge_osm_postgis_detections(
        self,
        topo_result: TopoAnalysisResult,
        osm_features: list[Any],
    ) -> TopoAnalysisResult:
        """Merges features detected by OSMPostGISDetector into TopoAnalysisResult.

        Args:
            topo_result: Existing TopoAnalysisResult from GIS shapefiles / topo data.
            osm_features: List of DetectedOSMFeature from OSMPostGISDetector.

        Returns:
            Updated TopoAnalysisResult with PostGIS-detected structures merged.
        """
        if not osm_features:
            return topo_result

        existing_structs = list(topo_result.structures)
        existing_chs = [s.chainage_m for s in existing_structs]
        added_count = 0

        for feat in osm_features:
            # Skip level crossings — they constrain gradient but are not bridge structures
            if feat.proposed_structure == "LEVEL_CROSSING":
                continue
            # Avoid duplicate structure proposals within 30m of existing structures
            if any(abs(feat.chainage_m - ex_ch) < 30.0 for ex_ch in existing_chs):
                continue

            new_struct = ProposedStructure(
                chainage_m=feat.chainage_m,
                structure_type=feat.proposed_structure,
                structure_name=feat.feature_name,
                span_config=feat.span_config,
                crossing_feature=feat.feature_name,
                clearance_req_m=feat.clearance_req_m,
                metadata={"source": "OSM PostGIS", "osm_id": feat.osm_id}
            )
            existing_structs.append(new_struct)
            existing_chs.append(feat.chainage_m)
            added_count += 1

        existing_structs.sort(key=lambda s: s.chainage_m)

        records = [
            {
                "chainage_m": s.chainage_m,
                "chainage_str": format_chainage(s.chainage_m),
                "structure_type": s.structure_type,
                "structure_name": s.structure_name,
                "span_config": s.span_config,
                "crossing_feature": s.crossing_feature,
                "clearance_req_m": s.clearance_req_m,
                "source": s.metadata.get("source", "GIS Topomap"),
            }
            for s in existing_structs
        ]

        merged_df = pd.DataFrame(records)
        self.logger.info(
            "Merged %d OSM PostGIS detected structure(s) into Bridge Schedule.",
            added_count
        )

        return TopoAnalysisResult(
            structures_df=merged_df,
            infringements_df=topo_result.infringements_df,
            structures=existing_structs,
            metadata={**topo_result.metadata, "osm_postgis_merged": True, "osm_pg_added": added_count}
        )
