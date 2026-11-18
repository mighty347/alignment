"""
Module: modules/dem_reader.py
Purpose: Module 3 — DEM Reader and Mathematical Interpolator.

Engineering Basis:
    Digital Elevation Models (DEM) represent topographic terrain. To extract
    Existing Ground Level (EGL) for our chainage points, we must:
    1. Ensure the DEM and chainage points share the exact same Euclidean space (UTM).
    2. Handle continuous sub-pixel sampling using bilinear interpolation, avoiding
       stepped artifacts characteristic of nearest-neighbor raster sampling.

Input:  GeoTIFF DEM file path, target CRS (from Module 1).
Output: DEMResult (reprojected DEM path + scipy interpolator).

Standards: RDSO / Indian Railways BG
Author   : Railway ESP Generator
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import rasterio
from rasterio.warp import Resampling, calculate_default_transform, reproject

from utils.logger import get_logger
from utils.validators import validate_geotiff

logger = get_logger(__name__)


class DEMReadError(Exception):
    """Custom exception for DEM Reader errors."""
    pass


@dataclass
class DEMResult:
    """Result object returned by DEMReader.

    Attributes:
        dataset_path: Path to the validated, UTM-projected DEM GeoTIFF.
        crs: Coordinate Reference System of the output DEM (e.g., 'EPSG:32643').
        nodata: The value representing missing data in the DEM.
        resolution: Tuple of (x_res, y_res) in meters.
        bounds: Tuple of (left, bottom, right, top).
        xs: Raw 1D array of column coords
        ys: Raw 1D array of row coords
        data: Raw 2D elevation matrix
        metadata: Engineering metadata dictionary.
    """
    dataset_path: Path
    crs: str
    nodata: float | None
    resolution: tuple[float, float]
    bounds: tuple[float, float, float, float]
    xs: np.ndarray
    ys: np.ndarray
    data: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


class DEMReader:
    """Read a GeoTIFF DEM, reproject if necessary, and build an interpolator.

    Implements Module 3 of the ESP Generator pipeline.

    Args:
        config: Configuration dictionary loaded from config.yaml.

    Example:
        >>> dem_result = DEMReader(config).read('data/srtm.tif', target_crs='EPSG:32643')
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        self.logger.info("Initialized %s", self.__class__.__name__)
        
        # Setup outputs directory for reprojected DEM
        self.outputs_dir = Path(config.get("paths", {}).get("outputs_dir", "outputs"))
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

    def read(self, dem_path: str | Path, target_crs: str) -> DEMResult:
        """Read DEM, ensure it matches target CRS, and build interpolator.

        Args:
            dem_path: Path to the input GeoTIFF DEM.
            target_crs: EPSG string of the required UTM zone (e.g., 'EPSG:32643').

        Returns:
            DEMResult containing the ready-to-use interpolator.

        Raises:
            DEMReadError: If the file is invalid or reading fails.
        """
        self.logger.info("Starting DEM reading process...")
        try:
            path = validate_geotiff(dem_path)
            
            # 1. Check current CRS and reproject if necessary
            working_dem_path = self._ensure_crs(path, target_crs)
            
            # 2. Extract point cloud data
            xs, ys, data, meta = self._extract_grid_arrays(working_dem_path)
            
            result = DEMResult(
                dataset_path=working_dem_path,
                crs=target_crs,
                nodata=meta["nodata"],
                resolution=meta["resolution"],
                bounds=meta["bounds"],
                xs=xs,
                ys=ys,
                data=data,
                metadata={
                    "original_file": str(path),
                    "reprojected_file": str(working_dem_path),
                    "width": meta["width"],
                    "height": meta["height"]
                }
            )
            
            self.logger.info("DEM reading complete. Ready for sampling.")
            return result
            
        except Exception as e:
            self.logger.error("DEM processing failed: %s", str(e))
            raise DEMReadError(f"DEM processing failed: {e}") from e

    def _ensure_crs(self, source_path: Path, target_crs: str) -> Path:
        """Check if DEM CRS matches target; if not, reproject it.
        
        Args:
            source_path: Path to the source DEM.
            target_crs: Required EPSG string.
            
        Returns:
            Path to the UTM-projected DEM (either original or newly generated).
        """
        with rasterio.open(source_path) as src:
            src_crs = src.crs
            if src_crs is not None and src_crs.to_string().upper() == target_crs.upper():
                self.logger.info("DEM CRS matches target (%s). No reprojection needed.", target_crs)
                return source_path
            
            self.logger.info("DEM CRS (%s) differs from target (%s). Reprojecting...", src_crs, target_crs)
            
            # Calculate transform and shape for the reprojected raster
            transform, width, height = calculate_default_transform(
                src.crs, target_crs, src.width, src.height, *src.bounds
            )
            
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': target_crs,
                'transform': transform,
                'width': width,
                'height': height
            })
            
            output_path = self.outputs_dir / f"{source_path.stem}_utm.tif"
            
            with rasterio.open(output_path, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=target_crs,
                        resampling=Resampling.bilinear
                    )
            
            self.logger.info("Reprojection complete. Saved to: %s", output_path)
            return output_path

    def _extract_grid_arrays(self, dem_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """
        Extract the X, Y coordinates and Z elevations for the raster grid.
        For large DEMs (> 50 million pixels), metadata is extracted lazily to prevent RAM overflow.
        """
        self.logger.debug("Extracting raw point coordinates for TIN generation.")
        
        # Suppress rasterio NotGeoreferencedWarning if any
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", rasterio.errors.NotGeoreferencedWarning)
            with rasterio.open(dem_path) as src:
                width, height = src.width, src.height
                total_pixels = width * height
                nodata = src.nodata
                
                meta = {
                    "nodata": nodata,
                    "resolution": src.res,
                    "bounds": src.bounds,
                    "width": width,
                    "height": height,
                    "total_pixels": total_pixels,
                    "is_lazy": False,
                }

                # High resolution / Massive DEM check (> 50 Million pixels)
                # Avoid reading 100+ GB full matrix into RAM; mark as lazy for windowed sampling in ElevationSampler
                if total_pixels > 50_000_000:
                    self.logger.info(
                        "Massive DEM detected (%d x %d = %.2f Billion pixels). "
                        "Using Lazy Windowed Sampling to optimize RAM usage.",
                        width, height, total_pixels / 1e9
                    )
                    meta["is_lazy"] = True
                    return np.array([]), np.array([]), np.empty((0, 0)), meta

                data = src.read(1).astype(float)
                
                # Mask out NoData values as NaN
                if nodata is not None:
                    data = np.where(np.isclose(data, nodata) | (data == nodata), np.nan, data)
                
                rows = np.arange(height)
                cols = np.arange(width)
                
                xs, _ = rasterio.transform.xy(src.transform, [0] * width, cols)
                _, ys = rasterio.transform.xy(src.transform, rows, [0] * height)
                
                ys_asc = np.array(ys)[::-1]
                xs_asc = np.array(xs)
                data_asc = data[::-1, :]
                
                self.logger.info("Extracted %d x %d grid points successfully.", width, height)
                return xs_asc, ys_asc, data_asc, meta
