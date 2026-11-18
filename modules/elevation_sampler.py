"""
Module: modules/elevation_sampler.py
Purpose: Module 4 — Elevation Sampler.

Engineering Basis:
    Samples the Existing Ground Level (EGL) for every chainage point
    along the railway alignment using the DEM interpolator.

Input: ChainageResult (Module 2), DEMResult (Module 3).
Output: EGLResult (DataFrame with new 'elevation_m' column).

Standards: RDSO / Indian Railways BG
Author   : Railway ESP Generator
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from modules.generate_chainage import ChainageResult
from modules.dem_reader import DEMResult
from utils.logger import get_logger


logger = get_logger(__name__)


class ElevationSampleError(Exception):
    """Custom exception for Elevation Sampler errors."""
    pass


@dataclass
class EGLResult:
    """Result object returned by ElevationSampler.

    Attributes:
        dataframe: Pandas DataFrame containing the chainage points and EGL.
            Columns: chainage_m, chainage_str, easting, northing, 
                     latitude, longitude, elevation_m
        metadata: Engineering metadata dictionary.
    """
    dataframe: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)


class ElevationSampler:
    """Samples DEM elevations at exactly specified chainage coordinates.

    Implements Module 4 of the ESP Generator pipeline.

    Args:
        config: Configuration dictionary loaded from config.yaml.

    Example:
        >>> egl_result = ElevationSampler(config).sample(chainage_result, dem_result)
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        self.logger.info("Initialized %s", self.__class__.__name__)

    def sample(
        self, 
        chainage_result: ChainageResult, 
        dem_result: DEMResult
    ) -> EGLResult:
        """Sample elevations for all points in the chainage result.

        Args:
            chainage_result: Output from Module 2 containing coordinates.
            dem_result: Output from Module 3 containing the interpolator.

        Returns:
            EGLResult containing updated DataFrame with 'elevation_m'.

        Raises:
            ElevationSampleError: If sampling fails mathematically.
        """
        self.logger.info("Starting elevation sampling...")
        
        try:
            df = chainage_result.dataframe.copy()
            
            if len(df) == 0:
                raise ElevationSampleError("Chainage DataFrame is empty.")
                
            if "easting" not in df.columns or "northing" not in df.columns:
                raise ElevationSampleError("Chainage DataFrame missing easting/northing columns.")
                
            # Extract coordinates
            eastings = df["easting"].values
            northings = df["northing"].values
            
            is_lazy = dem_result.metadata.get("is_lazy", False) or len(dem_result.xs) == 0

            # 1. Bounding box cropping (Buffer by 200m)
            min_e, max_e = float(np.min(eastings) - 200.0), float(np.max(eastings) + 200.0)
            min_n, max_n = float(np.min(northings) - 200.0), float(np.max(northings) + 200.0)

            self.logger.info("Cropping DEM points to alignment bounds: E[%.2f, %.2f] N[%.2f, %.2f]", min_e, max_e, min_n, max_n)

            if is_lazy:
                import rasterio
                import rasterio.windows
                from pyproj import Transformer

                with rasterio.open(dem_result.dataset_path) as src:
                    dem_crs = src.crs
                    # Auto-detect alignment CRS from coordinate magnitudes.
                    # UTM eastings are 100,000–999,999 m; WGS84 longitudes are -180–180.
                    # If eastings > 1000 the chainage points are in UTM metres.
                    if float(eastings[0]) > 1000.0:
                        # Infer UTM zone from easting magnitude and DEM centroid
                        # UTM zone number embedded in EPSG:326xx (Northern hemisphere)
                        mean_lon_approx = dem_result.metadata.get("mean_longitude",
                                          dem_result.metadata.get("centroid_lon", None))
                        if mean_lon_approx is not None:
                            zone = int((float(mean_lon_approx) + 180.0) / 6.0) + 1
                        else:
                            zone = 45  # Default: Zone 45N covers 84°–90°E
                        alignment_crs = f"EPSG:{32600 + zone}"
                    else:
                        alignment_crs = str(dem_crs)   # already in DEM CRS
                    self.logger.info("Alignment CRS inferred: %s", alignment_crs)

                    # Project alignment bbox from UTM → DEM native CRS (usually WGS84)
                    if str(dem_crs) != alignment_crs:
                        t = Transformer.from_crs(alignment_crs, dem_crs, always_xy=True)
                        # Transform all four corners to handle any projection distortion
                        corner_xs, corner_ys = t.transform(
                            [min_e, max_e, min_e, max_e],
                            [min_n, min_n, max_n, max_n],
                        )
                        bbox_left   = float(min(corner_xs))
                        bbox_right  = float(max(corner_xs))
                        bbox_bottom = float(min(corner_ys))
                        bbox_top    = float(max(corner_ys))
                    else:
                        bbox_left, bbox_right   = min_e, max_e
                        bbox_bottom, bbox_top   = min_n, max_n

                    # Clamp to DEM file extent
                    left   = max(src.bounds.left,   bbox_left)
                    right  = min(src.bounds.right,  bbox_right)
                    bottom = max(src.bounds.bottom, bbox_bottom)
                    top    = min(src.bounds.top,    bbox_top)

                    if left >= right or bottom >= top:
                        raise ElevationSampleError(
                            "Alignment bounding box does not overlap DEM extent. "
                            f"Alignment bbox: [{bbox_left:.4f},{bbox_bottom:.4f}] → "
                            f"[{bbox_right:.4f},{bbox_top:.4f}]  |  "
                            f"DEM extent: {src.bounds}"
                        )

                    window = rasterio.windows.from_bounds(left, bottom, right, top, src.transform)
                    win_h = int(round(window.height))
                    win_w = int(round(window.width))
                    pixel_count = win_h * win_w
                    max_pixels  = 500_000_000  # 500 M pixels approx 2 GB for float32

                    self.logger.info(
                        "DEM subset window: %d x %d pixels  (%.1f M pixels)",
                        win_w, win_h, pixel_count / 1_000_000,
                    )

                    if pixel_count > max_pixels:
                        # -- Fallback: point-by-point rasterio.sample() ----------------
                        self.logger.warning(
                            "DEM subset too large (%d M pixels). "
                            "Switching to point-by-point sampling.",
                            pixel_count // 1_000_000,
                        )
                        # Project sample points from UTM → DEM native CRS
                        if str(dem_crs) != alignment_crs:
                            px_lons, px_lats = t.transform(eastings.tolist(), northings.tolist())
                            sample_coords = list(zip(px_lons, px_lats))
                        else:
                            sample_coords = list(zip(eastings.tolist(), northings.tolist()))

                        nodata = src.nodata
                        raw_vals = np.array(
                            [v[0] for v in src.sample(sample_coords, indexes=1)],
                            dtype=float,
                        )
                        if nodata is not None:
                            raw_vals = np.where(
                                np.isclose(raw_vals, nodata) | (raw_vals == nodata),
                                np.nan, raw_vals,
                            )

                        # Preserve valid zero elevations and interpolate only gaps
                        # bracketed by valid samples. Trailing NaNs (including
                        # out-of-bounds samples) must not inherit the last height.
                        elev_series = pd.Series(raw_vals)
                        elevations = elev_series.interpolate(
                            method="linear", limit_area="inside"
                        ).values

                        nan_count = int(np.sum(np.isnan(elevations)))
                        df["elevation_m"] = elevations
                        meta = {
                            "total_points":    len(df),
                            "sampled_points":  len(df) - nan_count,
                            "missing_points":  nan_count,
                            "min_elevation":   float(np.nanmin(elevations)) if nan_count < len(df) else float("nan"),
                            "max_elevation":   float(np.nanmax(elevations)) if nan_count < len(df) else float("nan"),
                            "crs":             dem_result.crs,
                        }
                        self.logger.info(
                            "Point-sample complete. Valid: %d, Missing: %d, "
                            "EGL Range: [%.2f, %.2f] m",
                            meta["sampled_points"], meta["missing_points"],
                            meta["min_elevation"], meta["max_elevation"],
                        )
                        return EGLResult(dataframe=df, metadata=meta)

                    # ── Normal path: read the cropped window ──────────────────
                    subset_data = src.read(1, window=window).astype(float)
                    nodata = src.nodata
                    if nodata is not None:
                        subset_data = np.where(
                            np.isclose(subset_data, nodata) | (subset_data == nodata),
                            np.nan, subset_data,
                        )

                    win_transform = rasterio.windows.transform(window, src.transform)
                    h_win, w_win = subset_data.shape
                    rows_win = np.arange(h_win)
                    cols_win = np.arange(w_win)

                    xs_win, _ = rasterio.transform.xy(win_transform, [0] * w_win, cols_win)
                    _, ys_win = rasterio.transform.xy(win_transform, rows_win, [0] * h_win)

                    # Project DEM grid coords from DEM CRS → alignment UTM for interpolation
                    if str(dem_crs) != alignment_crs:
                        t_inv = Transformer.from_crs(dem_crs, alignment_crs, always_xy=True)
                        xs_utm, _ = t_inv.transform(xs_win, [ys_win[0]] * len(xs_win))
                        _, ys_utm = t_inv.transform([xs_win[0]] * len(ys_win), ys_win)
                        subset_x = np.array(xs_utm)
                        subset_y = np.array(ys_utm)[::-1]
                    else:
                        subset_x = np.array(xs_win)
                        subset_y = np.array(ys_win)[::-1]

                    subset_data = subset_data[::-1, :]

            else:
                x_mask = (dem_result.xs >= min_e) & (dem_result.xs <= max_e)
                y_mask = (dem_result.ys >= min_n) & (dem_result.ys <= max_n)

                subset_data = dem_result.data[y_mask, :][:, x_mask]
                subset_x = dem_result.xs[x_mask]
                subset_y = dem_result.ys[y_mask]
            
            if len(subset_x) < 2 or len(subset_y) < 2:
                raise ElevationSampleError("Not enough valid DEM grid points in the bounding box for interpolation.")

            self.logger.info("Performing 2D regular grid bilinear interpolation for %d stations...", len(eastings))

            # 2. Direct 2D Regular Grid Interpolation (Y = Northing, X = Easting)
            interpolator = RegularGridInterpolator(
                (subset_y, subset_x),
                subset_data,
                method="linear",
                bounds_error=False,
                fill_value=np.nan,
            )

            # Query coordinates as (northing, easting) matching grid dimensions (Y, X)
            pts = np.column_stack((northings, eastings))
            elevations = interpolator(pts)
            
            # Preserve valid zero elevations and interpolate only internal nodata
            # gaps. Out-of-bounds/trailing samples must remain NaN so downstream
            # design cannot silently reuse the last in-bounds terrain value.
            elev_series = pd.Series(elevations)
            elevations = elev_series.interpolate(
                method="linear", limit_area="inside"
            ).values
            
            nan_count = int(np.sum(np.isnan(elevations)))
            
            # Add to DataFrame
            df["elevation_m"] = elevations
            
            # Collect metadata
            meta = {
                "total_points": len(df),
                "sampled_points": len(df) - nan_count,
                "missing_points": nan_count,
                "min_elevation": float(np.nanmin(elevations)) if nan_count < len(df) else float("nan"),
                "max_elevation": float(np.nanmax(elevations)) if nan_count < len(df) else float("nan"),
                "crs": dem_result.crs
            }
            
            self.logger.info(
                "Elevation sampling complete. Valid: %d, Missing: %d, EGL Range: [%.2f, %.2f] m",
                meta["sampled_points"], meta["missing_points"], 
                meta["min_elevation"], meta["max_elevation"]
            )
            
            return EGLResult(dataframe=df, metadata=meta)
            
        except Exception as e:
            self.logger.error("Elevation sampling failed: %s", str(e))
            raise ElevationSampleError(f"Elevation sampling failed: {e}") from e

    def export_csv(
        self,
        result: EGLResult,
        output_path: str | Path,
    ) -> Path:
        """Export the EGL DataFrame to CSV.

        Args:
            result: EGLResult from sample().
            output_path: Destination CSV file path.

        Returns:
            Resolved path to the written CSV file.
        """
        out = Path(output_path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        result.dataframe.to_csv(out, index=False, float_format="%.6f")
        self.logger.info("EGL data exported to CSV: %s", out)
        return out
