"""
Module: modules/longitudinal_profile.py
Purpose: Module 5 — Longitudinal Profile (L-Section) Plotter.

Engineering Basis:
    Plots the Existing Ground Level (EGL) against Chainage. This is the 
    foundational 2D view for vertical alignment design. Uses matplotlib 
    to create an engineering-style graph with exaggerated vertical scale.

Input: EGLResult (Module 4).
Output: ProfileResult (Contains path to generated plot).

Standards: RDSO / Indian Railways BG
Author   : Railway ESP Generator
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, AutoMinorLocator

from modules.elevation_sampler import EGLResult
from utils.geo_utils import format_chainage
from utils.logger import get_logger


logger = get_logger(__name__)


class ProfilePlotError(Exception):
    """Custom exception for Profile rendering errors."""
    pass


@dataclass
class ProfileResult:
    """Result object returned by LongitudinalProfile.

    Attributes:
        dataframe: The original Pandas DataFrame passed from Module 4.
        plot_path_png: Path to the generated PNG plot.
        plot_path_pdf: Path to the generated PDF plot (if generated).
        metadata: Engineering metadata dictionary.
    """
    dataframe: pd.DataFrame
    plot_path_png: Path
    plot_path_pdf: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class LongitudinalProfile:
    """Renders the engineering L-Section profile of the terrain.

    Implements Module 5 of the ESP Generator pipeline.

    Args:
        config: Configuration dictionary loaded from config.yaml.

    Example:
        >>> profile_result = LongitudinalProfile(config).plot(egl_result)
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        self.logger.info("Initialized %s", self.__class__.__name__)
        
        self.outputs_dir = Path(config.get("paths", {}).get("outputs_dir", "outputs"))
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        
        # Plotting configs
        # Default size is very wide (e.g., 24x6 inches) to simulate a long profile roll
        self.fig_width = 24.0
        self.fig_height = 6.0

    def plot(self, egl_result: EGLResult) -> ProfileResult:
        """Plot the longitudinal profile from chainage and elevation data.

        Args:
            egl_result: Output from Module 4 containing 'chainage_m' and 'elevation_m'.

        Returns:
            ProfileResult containing paths to the rendered files.

        Raises:
            ProfilePlotError: If plotting fails due to bad data.
        """
        self.logger.info("Starting Longitudinal Profile generation...")
        
        try:
            df = egl_result.dataframe
            
            if "chainage_m" not in df.columns or "elevation_m" not in df.columns:
                raise ProfilePlotError("Required columns (chainage_m, elevation_m) not found in EGLResult.")

            # Filter out NaNs for the continuous plot
            valid_mask = df["elevation_m"].notna()
            chainage_valid = df.loc[valid_mask, "chainage_m"].values
            elevation_valid = df.loc[valid_mask, "elevation_m"].values
            
            if len(chainage_valid) < 2:
                raise ProfilePlotError("Not enough valid elevation points to plot a profile.")

            # ---------------------------------------------------------
            # Matplotlib Setup
            # ---------------------------------------------------------
            fig, ax = plt.subplots(figsize=(self.fig_width, self.fig_height), dpi=150)
            
            # Plot EGL line
            ax.plot(
                chainage_valid, 
                elevation_valid, 
                color="saddlebrown", 
                linewidth=1.5, 
                label="Existing Ground Level (EGL)"
            )
            
            # Fill under the terrain line for a solid earth look
            ax.fill_between(
                chainage_valid, 
                elevation_valid, 
                y2=np.min(elevation_valid) - 10.0, 
                color="saddlebrown", 
                alpha=0.1
            )

            # ---------------------------------------------------------
            # Axis formatting (Engineering Style)
            # ---------------------------------------------------------
            ax.set_title("Longitudinal Profile (L-Section)", fontsize=14, weight="bold")
            ax.set_xlabel("Chainage", fontsize=11, weight="bold")
            ax.set_ylabel("Elevation (m AMSL)", fontsize=11, weight="bold")
            
            # Y-axis scaling (buffer 5m above and below)
            y_min = np.min(elevation_valid)
            y_max = np.max(elevation_valid)
            ax.set_ylim(y_min - 5.0, y_max + 15.0)  # Extra room on top for legend
            
            # X-axis scaling
            ax.set_xlim(np.min(chainage_valid), np.max(chainage_valid))
            
            # Formatter for X-axis (convert float meters to Km X+YYY)
            def _chainage_formatter(x, pos):
                return format_chainage(x)
            
            ax.xaxis.set_major_formatter(FuncFormatter(_chainage_formatter))
            
            # Grid lines
            ax.grid(True, which='major', color='black', linewidth=0.5, alpha=0.6)
            ax.grid(True, which='minor', color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
            
            # Add minor ticks for the graph paper effect
            ax.xaxis.set_minor_locator(AutoMinorLocator(4))
            ax.yaxis.set_minor_locator(AutoMinorLocator(5))
            
            ax.legend(loc="upper right")
            plt.tight_layout()

            # ---------------------------------------------------------
            # Save Outputs
            # ---------------------------------------------------------
            png_path = self.outputs_dir / "longitudinal_profile.png"
            pdf_path = self.outputs_dir / "longitudinal_profile.pdf"
            
            try:
                fig.savefig(png_path, format="png", bbox_inches="tight")
                fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
                self.logger.info("Longitudinal Profile saved to: %s", png_path)
            except Exception as e_plot:
                self.logger.warning("Could not save matplotlib plot: %s", e_plot)
            finally:
                plt.close(fig)

            meta = {
                "png_path": str(png_path),
                "pdf_path": str(pdf_path),
                "min_elevation_plotted": float(y_min),
                "max_elevation_plotted": float(y_max),
                "total_points_plotted": len(chainage_valid)
            }

            return ProfileResult(
                dataframe=df,
                plot_path_png=png_path,
                plot_path_pdf=pdf_path,
                metadata=meta
            )

        except Exception as e:
            self.logger.error("Profile plotting failed: %s", str(e))
            raise ProfilePlotError(f"Profile plotting failed: {e}") from e
