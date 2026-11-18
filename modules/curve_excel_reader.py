"""
Module: modules/curve_excel_reader.py
Purpose: Indian Railways Proposed Curve Schedule Excel Reader (Prop.Curve List).

Parses pre-calculated horizontal curve schedules from Excel files (e.g., DPR "Prop.Curve List"
tables or individual curve calculation sheets) and constructs a fully populated HGeomResult.
This allows the drafting engines (CompositeDrawer, PlanDrawer, LSectionDrawer) to render exact
curve arcs, transition spirals, parameter boxes, and L-Section data bands without recalculating.

Supported Excel Table Format ("Prop.Curve List"):
    Col 1:  Sr. NO. / Curve No.
    Col 2:  Delta / Deflection Angle (deg)
    Col 3:  Radius (m) (e.g. '700.000m', 700)
    Col 4:  Degree (D = 1750/R)
    Col 5:  TPTC-1 / T1 / TS (e.g. '0+991.21m')
    Col 6:  TPCC-1 / J1 / SC (e.g. '1+111.21m')
    Col 7:  TPCC-2 / J2 / CS (e.g. '1+261.34m')
    Col 8:  TPTC-2 / T2 / ST (e.g. '1+381.34m')
    Col 9:  Transition Length (m)
    Col 10: Circular Curve length (m) (TPCC-2 - TPCC-1)
    Col 11: Total Circular length (m) (TPTC-2 - TPTC-1)
    Col 12: Tangent Length (m)
    Col 13: Cant (Ca) (mm)
    Col 14: Speed Potential / Vmax (km/h)

Standards: RDSO GE:G-1 / Indian Railways Permanent Way Manual (IRPWM Para 405)
Author   : Railway ESP Generator
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path
from typing import Any, Optional, Union

# Add project root directory to sys.path for direct module execution
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
from shapely.geometry import LineString, Point

from modules.horizontal_geometry import HGeomResult
from modules.read_kml import AlignmentResult
from utils.geo_utils import format_chainage
from utils.logger import get_logger

logger = get_logger(__name__)


class CurveExcelReaderError(Exception):
    """Custom exception raised for Curve Excel Reader errors."""
    pass


def parse_chainage_str(val: Any) -> float:
    """Parse chainage representation into float chainage in meters.

    Examples:
        - '0+991.21m' -> 991.21
        - '1+111.21m' -> 1111.21
        - '10+513.54m' -> 10513.54
        - 'Km 002+473.61' -> 2473.61
        - 2631.76 -> 2631.76
    """
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)

    s = str(val).strip()
    # Remove 'km' or 'km.' prefix/suffix first (case-insensitive)
    s = re.sub(r"(?i)\bkm\b", "", s).strip()
    # Remove 'm' or 'mtr' suffix
    s = re.sub(r"(?i)[a-z]+$", "", s).strip()

    # Match '1+111.21' or '0+991.21'
    if "+" in s:
        parts = s.split("+")
        if len(parts) == 2:
            try:
                km = float(parts[0].strip())
                m = float(parts[1].strip())
                return km * 1000.0 + m
            except ValueError:
                pass

    # Direct float conversion fallback
    try:
        return float(s.replace(",", ""))
    except ValueError:
        logger.warning("Could not parse chainage value '%s', returning 0.0", val)
        return 0.0


def parse_radius_value(val: Any) -> float:
    """Parse curve radius value in meters, handling trailing 'm', commas, etc."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip().replace("m", "").replace("M", "").replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return 0.0


def parse_angle_value(val: Any) -> float:
    """Parse angle value in decimal degrees, DMS strings (e.g. 1° 33' 31")."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip().replace(",", "")
    # Check for DMS format e.g. 1° 33' 31" or 1 O 33 ' 31 "
    dms_match = re.search(r"(\d+)\s*[°OodD]\s*(\d+)\s*['’]\s*([\d\.]+)", s)
    if dms_match:
        deg = float(dms_match.group(1))
        minute = float(dms_match.group(2))
        sec = float(dms_match.group(3))
        return deg + minute / 60.0 + sec / 3600.0
    try:
        return float(s)
    except ValueError:
        return 0.0


class CurveExcelReader:
    """Reads Indian Railways Proposed Curve Schedule Excel sheets and creates HGeomResult."""

    def __init__(self, config: Optional[dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.logger = get_logger(self.__class__.__name__)
        self.logger.info("Initialized %s", self.__class__.__name__)

    def read_curve_schedule(
        self,
        excel_path: Union[str, Path, pd.DataFrame],
        alignment_result: Optional[AlignmentResult] = None,
        sheet_name: Optional[Union[str, int]] = 0,
    ) -> HGeomResult:
        """Parse curve list from Excel file path or existing DataFrame.

        Args:
            excel_path: Path to .xlsx/.xls file, or pandas DataFrame.
            alignment_result: Optional AlignmentResult from Module 1 to interpolate
                true 2D (Easting, Northing) coordinates along centerline.
            sheet_name: Excel sheet name or index. Defaults to first sheet (0).

        Returns:
            HGeomResult populated with parsed curve records and engineering metadata.
        """
        self.logger.info("Reading curve schedule from: %s", excel_path)

        if isinstance(excel_path, pd.DataFrame):
            raw_df = excel_path.copy()
        else:
            path = Path(excel_path)
            if not path.exists():
                raise FileNotFoundError(f"Curve Excel schedule file not found at: {path.resolve()}")

            # Determine whether file is a tabular schedule or multi-sheet curve workbook
            excel_file = pd.ExcelFile(path)
            sheet_names = excel_file.sheet_names
            self.logger.info("Found sheets in Excel: %s", sheet_names)

            # Check if there is a 'Prop.Curve List' or similar sheet
            target_sheet: Union[str, int] = sheet_name or 0
            for name in sheet_names:
                if any(k in name.lower() for k in ["prop.curve", "prop_curve", "curve list", "curvelist", "curves"]):
                    target_sheet = name
                    break

            raw_df = pd.read_excel(path, sheet_name=target_sheet, header=None)

        return self._parse_curve_dataframe(raw_df, alignment_result)

    def _parse_curve_dataframe(
        self,
        raw_df: pd.DataFrame,
        alignment_result: Optional[AlignmentResult] = None,
    ) -> HGeomResult:
        """Normalize table header rows and construct HGeomResult records."""
        # Find the header row containing 'radius', 'delta', 'tptc', or 'sr. no'
        header_row_idx = -1
        for idx in range(min(15, len(raw_df))):
            row_str = " ".join([str(v).lower() for v in raw_df.iloc[idx].values if pd.notna(v)])
            if ("radius" in row_str or "delta" in row_str) and ("tptc" in row_str or "tangent" in row_str or "sr." in row_str or "curve" in row_str):
                header_row_idx = idx
                break

        if header_row_idx != -1:
            headers = [str(col).strip() if pd.notna(col) else f"col_{i}" for i, col in enumerate(raw_df.iloc[header_row_idx])]
            df = raw_df.iloc[header_row_idx + 1:].copy()
            df.columns = headers
        else:
            df = raw_df.copy()

        # Clean numeric indexing row if present (e.g. row with 1, 2, 3, ... 14)
        if not df.empty:
            first_row_vals = [str(v).strip() for v in df.iloc[0].values if pd.notna(v)]
            if first_row_vals and all(v.isdigit() for v in first_row_vals[:4]):
                df = df.iloc[1:].copy()

        self.logger.info("Identified table columns: %s", list(df.columns))

        # Build column index mapping
        col_map = self._map_columns(list(df.columns))
        self.logger.info("Mapped curve schedule columns: %s", col_map)

        ip_records: list[dict[str, Any]] = []
        alignment_line: Optional[LineString] = alignment_result.alignment_line if alignment_result else None

        for row_idx, (_, row) in enumerate(df.iterrows(), start=1):
            try:
                # Check for valid curve number / row
                sr_no_val = row.get(col_map.get("sr_no", "")) if col_map.get("sr_no") else row_idx
                if pd.isna(sr_no_val) or str(sr_no_val).strip() == "":
                    continue

                radius_val = parse_radius_value(row.get(col_map.get("radius", "")))
                if radius_val <= 0:
                    continue

                delta_deg = parse_angle_value(row.get(col_map.get("delta", "")))
                degree_val = parse_angle_value(row.get(col_map.get("degree", "")))
                if degree_val <= 0 and radius_val > 0:
                    degree_val = round(1750.0 / radius_val, 4)

                # Milestone chainages
                tptc1 = parse_chainage_str(row.get(col_map.get("tptc_1", "")))
                tpcc1 = parse_chainage_str(row.get(col_map.get("tpcc_1", "")))
                tpcc2 = parse_chainage_str(row.get(col_map.get("tpcc_2", "")))
                tptc2 = parse_chainage_str(row.get(col_map.get("tptc_2", "")))

                # Lengths
                ls_val = parse_radius_value(row.get(col_map.get("trans_len", "")))
                if ls_val <= 0 and tpcc1 > tptc1:
                    ls_val = round(tpcc1 - tptc1, 2)

                lc_val = parse_radius_value(row.get(col_map.get("circ_len", "")))
                if lc_val <= 0 and tpcc2 > tpcc1:
                    lc_val = round(tpcc2 - tpcc1, 2)

                cl_val = parse_radius_value(row.get(col_map.get("total_len", "")))
                if cl_val <= 0 and tptc2 > tptc1:
                    cl_val = round(tptc2 - tptc1, 2)

                tangent_len = parse_radius_value(row.get(col_map.get("tangent_len", "")))
                if tangent_len <= 0 and delta_deg > 0:
                    shift = (ls_val ** 2) / (24.0 * radius_val) if radius_val > 0 else 0.0
                    tangent_len = (ls_val / 2.0) + (radius_val + shift) * math.tan(math.radians(delta_deg) / 2.0)

                shift_val = (ls_val ** 2) / (24.0 * radius_val) if radius_val > 0 else 0.0

                cant_val = parse_radius_value(row.get(col_map.get("cant", "")))
                speed_val = parse_radius_value(row.get(col_map.get("speed", "")))
                if speed_val <= 0:
                    speed_val = 160.0

                # Apex / IP Chainage
                apex_chainage_m = tptc1 + tangent_len

                # Coordinates lookup from alignment polyline if available
                easting, northing = 0.0, 0.0
                if alignment_line is not None and not alignment_line.is_empty:
                    length = alignment_line.length
                    # Clamp apex chainage to alignment length
                    norm_ch = max(0.0, min(length, apex_chainage_m))
                    pt: Point = alignment_line.interpolate(norm_ch)
                    easting, northing = pt.x, pt.y

                # Auto-detect Turn Direction from Raw KML Alignment Geometry
                direction = "RHS"
                dir_col = col_map.get("direction")
                if dir_col and pd.notna(row.get(dir_col)):
                    d_str = str(row.get(dir_col)).strip().upper()
                    if "L" in d_str:
                        direction = "LHS"
                elif alignment_line is not None and not alignment_line.is_empty:
                    # Sample incoming and outgoing tangents across the curve
                    ch_s = max(0.0, tptc1)
                    ch_e = min(float(alignment_line.length), tptc2)
                    if ch_e > ch_s:
                        ch_m = (ch_s + ch_e) / 2.0
                        pt_s = alignment_line.interpolate(ch_s)
                        pt_m = alignment_line.interpolate(ch_m)
                        pt_e = alignment_line.interpolate(ch_e)

                        # 2D cross product: (v_in x v_out)
                        v_in_x, v_in_y = pt_m.x - pt_s.x, pt_m.y - pt_s.y
                        v_out_x, v_out_y = pt_e.x - pt_m.x, pt_e.y - pt_m.y
                        cross_z = (v_in_x * v_out_y) - (v_in_y * v_out_x)

                        # In UTM coordinates: cross_z > 0 is Left-Hand Turn (LHS), cross_z < 0 is Right-Hand Turn (RHS)
                        direction = "LHS" if cross_z > 0 else "RHS"

                try:
                    ip_num = int(float(str(sr_no_val).replace("C-", "").replace("C", "")))
                except ValueError:
                    ip_num = len(ip_records) + 1

                record = {
                    "ip_index": ip_num,
                    "easting": easting,
                    "northing": northing,
                    "chainage_m": round(apex_chainage_m, 2),
                    "chainage_str": format_chainage(apex_chainage_m),
                    "deflection_deg": delta_deg,
                    "deflection_angle_deg": abs(delta_deg),
                    "direction": direction,
                    "radius_m": radius_val,
                    "radius2_m": None,
                    "curve_type": "SIMPLE",
                    "degree_of_curve": degree_val,
                    "transition_length_m": ls_val,
                    "shift_m": round(shift_val, 4),
                    "tangent_length_m": round(tangent_len, 2),
                    "circular_curve_length_m": round(lc_val, 2),
                    "curve_length_m": round(cl_val, 2),
                    "tptc_chainage_m": round(tptc1, 2),
                    "tpcc_chainage_m": round(tpcc1, 2),
                    "ctpc_chainage_m": round(tpcc2, 2),
                    "ctpt_chainage_m": round(tptc2, 2),
                    "cant_mm": cant_val,
                    "actual_cd_mm": 75.0,
                    "cant_excess_mm": 0.0,
                    "permissible_speed_kmph": speed_val,
                }
                ip_records.append(record)

            except Exception as err:
                self.logger.warning("Skipping invalid curve row %d: %s", row_idx, err)

        df_ips = pd.DataFrame(ip_records)
        self.logger.info("Successfully parsed %d curve records from Excel schedule.", len(df_ips))

        metadata = {
            "source": "Excel Prop.Curve List",
            "total_ips": len(df_ips),
            "curves_parsed": len(df_ips),
        }

        return HGeomResult(ip_dataframe=df_ips, metadata=metadata)

    def _map_columns(self, columns: list[str]) -> dict[str, str]:
        """Maps varying column headers from Excel to standard internal field keys."""
        mapping: dict[str, str] = {}
        for col in columns:
            c = str(col).lower().replace("\n", " ").replace("\r", " ").strip()
            if "sr." in c or "sr no" in c or "sr_no" in c or "curve no" in c or "curve_no" in c or c == "sr":
                mapping.setdefault("sr_no", col)
            elif "delta" in c or "deflection" in c:
                mapping.setdefault("delta", col)
            elif "radius" in c:
                mapping.setdefault("radius", col)
            elif "degree" in c or "deg" in c:
                mapping.setdefault("degree", col)
            elif "tptc-1" in c or "tptc 1" in c or "tptc_1" in c or "t1" in c:
                mapping.setdefault("tptc_1", col)
            elif "tpcc-1" in c or "tpcc 1" in c or "tpcc_1" in c or "j1" in c:
                mapping.setdefault("tpcc_1", col)
            elif "tpcc-2" in c or "tpcc 2" in c or "tpcc_2" in c or "j2" in c:
                mapping.setdefault("tpcc_2", col)
            elif "tptc-2" in c or "tptc 2" in c or "tptc_2" in c or "t2" in c:
                mapping.setdefault("tptc_2", col)
            elif "transition" in c and "length" in c:
                mapping.setdefault("trans_len", col)
            elif "circular curve length" in c or ("circular" in c and "length" in c and "(7)-(6)" in c):
                mapping.setdefault("circ_len", col)
            elif "total" in c and "length" in c:
                mapping.setdefault("total_len", col)
            elif "tangent" in c and "length" in c:
                mapping.setdefault("tangent_len", col)
            elif "cant" in c or "ca" in c:
                mapping.setdefault("cant", col)
            elif "speed" in c or "kmph" in c or "potential" in c:
                mapping.setdefault("speed", col)
            elif "direction" in c or "side" in c or "hand" in c:
                mapping.setdefault("direction", col)

        # Fallback to positional mapping if headers were generic (e.g. col_0 .. col_13)
        if len(mapping) < 5 and len(columns) >= 14:
            self.logger.info("Applying positional column fallback for 14-column Prop.Curve List")
            pos_keys = [
                "sr_no", "delta", "radius", "degree",
                "tptc_1", "tpcc_1", "tpcc_2", "tptc_2",
                "trans_len", "circ_len", "total_len", "tangent_len",
                "cant", "speed"
            ]
            for idx, key in enumerate(pos_keys):
                if idx < len(columns):
                    mapping[key] = columns[idx]

        return mapping
