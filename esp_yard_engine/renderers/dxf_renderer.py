"""CAD DXF Renderer for Indian Railways Station Yard (ESP) Layouts.

Renders complete multi-track station yard layouts, turnouts, fouling marks,
CSR dimensions, platforms, FOBs, and station buildings onto AutoCAD DXF modelspace
strictly conforming to RDSO drafting conventions.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Optional

import ezdxf
from ezdxf import units
from ezdxf.enums import TextEntityAlignment

from esp_yard_engine.models.yard_schema import (
    YardConfig,
    load_yard_standards,
)
from esp_yard_engine.geometry.alignment_projector import AlignmentProjector
from esp_yard_engine.geometry.track_generator import (
    ProjectedYardLayout,
    generate_projected_tracks,
)
from esp_yard_engine.geometry.civil_generator import (
    ProjectedCivilLayout,
    generate_projected_civil_elements,
)

logger = logging.getLogger("ESP_Yard_Engine.DXFRenderer")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class YardDXFRenderer:
    """Renders complete 2D Station Yard Engineering Scale Plans into ezdxf modelspace."""

    def __init__(self, standards: Optional[dict[str, Any]] = None) -> None:
        self.standards = standards or load_yard_standards()
        self.layers_cfg = self.standards.get("cad_layers", {})

    def setup_layers(self, doc: ezdxf.document.Drawing) -> None:
        """Register official RDSO ESP layers in the DXF document."""
        for key, lay_info in self.layers_cfg.items():
            name = lay_info.get("name", f"ESP_YARD_{key}")
            color = int(lay_info.get("color_aci", 7))
            lw = int(float(lay_info.get("lineweight_mm", 0.25)) * 100)
            lt = str(lay_info.get("linetype", "Continuous"))

            if name not in doc.layers:
                doc.layers.add(name=name, color=color, lineweight=lw, linetype=lt)
            else:
                doc.layers.get(name).color = color

        # Standard text style
        if "ESP_STYLE" not in doc.styles:
            doc.styles.add("ESP_STYLE", font="simplex.shx")

    def _get_layer(self, key: str, fallback_color: int = 7) -> str:
        """Get layer name from standards configuration."""
        cfg = self.layers_cfg.get(key, {})
        return str(cfg.get("name", f"ESP_YARD_{key}"))

    def _get_layer_color(self, key: str, fallback_color: int = 7) -> int:
        """Get layer color ACI code from standards configuration."""
        cfg = self.layers_cfg.get(key, {})
        return int(cfg.get("color_aci", fallback_color))

    def render_yard(
        self,
        msp: Any,
        yard_tracks: ProjectedYardLayout,
        civil_layout: Optional[ProjectedCivilLayout] = None,
        anno_scale: float = 1.0,
        text_height_m: float = 2.5,
        render_flagpole: bool = True,
        render_text: bool = True,
        crossover_as_spline: bool = True,
        show_station_text_only: bool = False,
    ) -> None:
        """Render all yard tracks, turnouts, fouling marks, CSR lines, platforms, and civil structures.

        Args:
            msp: ezdxf modelspace to draw on.
            yard_tracks: Output from generate_projected_tracks().
            civil_layout: Output from generate_projected_civil_elements().
            anno_scale: Annotation scale multiplier (default: 1.0).
            text_height_m: Base text height in ground meters.
            render_flagpole: Whether to draw the station flagpole & callout box (set False if drawn by PlanDrawer).
            render_text: Whether to render text labels and dimension text (default: True). Set False to hide text.
            crossover_as_spline: Whether to render crossover / turnout curves as AutoCAD SPLINE entities (default: True).
            show_station_text_only: When True, only station name text is shown, hiding internal track/turnout/civil labels.
        """
        th = text_height_m * anno_scale

        if show_station_text_only:
            render_yard_labels = False
            render_station_callout = True
        else:
            render_yard_labels = render_text
            render_station_callout = render_text

        lay_main = self._get_layer("TRACK_MAIN")
        lay_loop = self._get_layer("TRACK_LOOP")
        lay_turnout = self._get_layer("TURNOUT")
        lay_platform = self._get_layer("PLATFORM")
        lay_struct = self._get_layer("STRUCTURES")
        lay_dim = self._get_layer("DIMENSIONS")
        lay_text = self._get_layer("TEXT")
        lay_fm = self._get_layer("FOULING_MARK")

        col_main = self._get_layer_color("TRACK_MAIN", 1)
        col_loop = self._get_layer_color("TRACK_LOOP", 1)
        col_turnout = self._get_layer_color("TURNOUT", 1)
        col_platform = self._get_layer_color("PLATFORM", 7)
        col_struct = self._get_layer_color("STRUCTURES", 7)

        # ── 1. Draw Track Centerlines ──
        for trk in yard_tracks.tracks:
            is_main = abs(trk.lateral_offset_m) < 0.01
            trk_lay = lay_main if is_main else lay_loop
            trk_col = col_main if is_main else col_loop
            if len(trk.polyline_world) >= 2:
                msp.add_lwpolyline(
                    trk.polyline_world,
                    dxfattribs={"layer": trk_lay, "color": trk_col},
                )

                # Track Name Label near middle
                if render_yard_labels:
                    mid_idx = len(trk.polyline_world) // 2
                    mid_pt = trk.polyline_world[mid_idx]
                    msp.add_text(
                        trk.name,
                        dxfattribs={
                            "layer": lay_text,
                            "height": th * 0.9,
                            "style": "ESP_STYLE",
                        },
                    ).set_placement((mid_pt[0], mid_pt[1] + 1.0 * anno_scale), align=TextEntityAlignment.BOTTOM_CENTER)

        # ── 2. Draw Turnouts / Crossovers (Points & Crossings) ──
        for to in yard_tracks.turnouts:
            if len(to.turnout_polyline_world) >= 2:
                if crossover_as_spline:
                    msp.add_spline(
                        to.turnout_polyline_world,
                        degree=3,
                        dxfattribs={"layer": lay_turnout, "color": col_turnout},
                    )
                else:
                    msp.add_lwpolyline(
                        to.turnout_polyline_world,
                        dxfattribs={"layer": lay_turnout, "color": col_turnout},
                    )

            # Switch Toe Marker & Point Number
            if render_yard_labels:
                toe_pt = to.switch_toe_world
                to_label = f"Pt {to.turnout_id} ({to.turnout_type.value.replace('_', ' ')})"
                msp.add_text(
                    to_label,
                    dxfattribs={
                        "layer": lay_text,
                        "height": th * 0.8,
                        "style": "ESP_STYLE",
                    },
                ).set_placement((toe_pt[0], toe_pt[1] + 1.5 * anno_scale), align=TextEntityAlignment.BOTTOM_CENTER)

            # Fouling Mark (FM) Cross Bar & Label
            b1, b2 = to.fouling_indicator_bar_world
            msp.add_line(b1, b2, dxfattribs={"layer": lay_fm})
            if render_yard_labels:
                msp.add_text(
                    "FM",
                    dxfattribs={
                        "layer": lay_fm,
                        "height": th * 0.75,
                        "style": "ESP_STYLE",
                    },
                ).set_placement((to.fouling_mark_world[0], to.fouling_mark_world[1] + 0.8 * anno_scale), align=TextEntityAlignment.BOTTOM_CENTER)

        # ── 3. Draw CSR Dimension Line between Outer Fouling Marks ──
        if render_yard_labels and len(yard_tracks.turnouts) >= 2:
            fm_pts = sorted([to.fouling_mark_world for to in yard_tracks.turnouts], key=lambda p: p[0])
            fm_west = fm_pts[0]
            fm_east = fm_pts[-1]

            # Dimension line between facing and trailing FMs
            dim_y = (fm_west[1] + fm_east[1]) / 2.0 - 2.5 * anno_scale
            p_d1 = (fm_west[0], dim_y)
            p_d2 = (fm_east[0], dim_y)

            msp.add_line(p_d1, p_d2, dxfattribs={"layer": lay_dim})
            # Arrow ticks at ends
            tick_h = 1.0 * anno_scale
            msp.add_line((p_d1[0], dim_y - tick_h), (p_d1[0], dim_y + tick_h), dxfattribs={"layer": lay_dim})
            msp.add_line((p_d2[0], dim_y - tick_h), (p_d2[0], dim_y + tick_h), dxfattribs={"layer": lay_dim})

            csr_text = f"<--- CSR = {yard_tracks.csr_length_m:.1f} m --->"
            mid_x = (p_d1[0] + p_d2[0]) / 2.0
            msp.add_text(
                csr_text,
                dxfattribs={
                    "layer": lay_dim,
                    "height": th * 0.85,
                    "style": "ESP_STYLE",
                },
            ).set_placement((mid_x, dim_y + 0.5 * anno_scale), align=TextEntityAlignment.BOTTOM_CENTER)

        # ── 4. Draw Safety Overruns / Sand Humps ──
        for hump in yard_tracks.overruns:
            if len(hump.polyline_world) >= 2:
                if crossover_as_spline:
                    msp.add_spline(
                        hump.polyline_world,
                        degree=3,
                        dxfattribs={"layer": lay_loop, "color": col_loop},
                    )
                else:
                    msp.add_lwpolyline(
                        hump.polyline_world,
                        dxfattribs={"layer": lay_loop, "color": col_loop},
                    )
            # Buffer Stop bar
            b1, b2 = hump.buffer_stop_bar_world
            msp.add_line(b1, b2, dxfattribs={"layer": lay_struct, "color": col_struct})
            if render_yard_labels:
                b_mid = ((b1[0] + b2[0]) / 2.0, (b1[1] + b2[1]) / 2.0)
                msp.add_text(
                    "BUFFER STOP",
                    dxfattribs={
                        "layer": lay_struct,
                        "height": th * 0.75,
                        "style": "ESP_STYLE",
                    },
                ).set_placement((b_mid[0], b_mid[1] + 1.0 * anno_scale), align=TextEntityAlignment.BOTTOM_CENTER)

        # ── 5. Draw Platforms (Civil Infrastructure) ──
        if civil_layout:
            for pf in civil_layout.platforms:
                # Closed boundary
                if len(pf.boundary_polygon_world) >= 3:
                    msp.add_lwpolyline(
                        pf.boundary_polygon_world,
                        close=True,
                        dxfattribs={"layer": lay_platform, "color": col_platform},
                    )
                # Coping line
                if len(pf.coping_edge_world) >= 2:
                    msp.add_lwpolyline(
                        pf.coping_edge_world,
                        dxfattribs={"layer": lay_platform, "color": col_platform},
                    )
                # End ramps
                if pf.facing_ramp_polygon_world:
                    msp.add_lwpolyline(
                        pf.facing_ramp_polygon_world,
                        close=True,
                        dxfattribs={"layer": lay_platform, "color": col_platform},
                    )
                if pf.trailing_ramp_polygon_world:
                    msp.add_lwpolyline(
                        pf.trailing_ramp_polygon_world,
                        close=True,
                        dxfattribs={"layer": lay_platform, "color": col_platform},
                    )

                # Platform Name Label
                if render_yard_labels:
                    msp.add_text(
                        f"{pf.name} (L={pf.length_m:.0f}m, W={pf.width_m:.1f}m)",
                        dxfattribs={
                            "layer": lay_text,
                            "height": th * 0.95,
                            "style": "ESP_STYLE",
                        },
                    ).set_placement(pf.label_pos_world, align=TextEntityAlignment.MIDDLE_CENTER)

            # ── 6. Draw Station Building ──
            if civil_layout.station_building:
                sb = civil_layout.station_building
                if len(sb.boundary_polygon_world) >= 4:
                    msp.add_lwpolyline(
                        sb.boundary_polygon_world,
                        close=True,
                        dxfattribs={"layer": lay_struct, "color": col_struct},
                    )
                if render_yard_labels:
                    msp.add_text(
                        sb.name,
                        dxfattribs={
                            "layer": lay_text,
                            "height": th * 1.05,
                            "style": "ESP_STYLE",
                        },
                    ).set_placement(sb.label_pos_world, align=TextEntityAlignment.MIDDLE_CENTER)

            # ── 7. Draw Foot Over Bridge (FOB) ──
            for fob in civil_layout.fobs:
                if len(fob.bridge_span_polygon_world) >= 4:
                    msp.add_lwpolyline(
                        fob.bridge_span_polygon_world,
                        close=True,
                        dxfattribs={"layer": lay_struct, "color": col_struct},
                    )
                for stair in fob.stair_polygons_world:
                    msp.add_lwpolyline(stair, close=True, dxfattribs={"layer": lay_struct, "color": col_struct})

                if render_yard_labels:
                    msp.add_text(
                        fob.name,
                        dxfattribs={
                            "layer": lay_text,
                            "height": th * 0.85,
                            "style": "ESP_STYLE",
                        },
                    ).set_placement(fob.center_pos_world, align=TextEntityAlignment.MIDDLE_CENTER)

        # ── 8. Draw Station Centerline Flagpole & Callout ──
        if render_flagpole and yard_tracks.station_marker:
            sm = yard_tracks.station_marker
            p_bot = sm.track_point_world
            p_top = sm.flagpole_top_world

            # Green dash-dot flagpole line
            msp.add_line(p_bot, p_top, dxfattribs={"layer": lay_text})

            # Masked 3-line station callout box
            if render_station_callout:
                km_val = sm.chainage_m / 1000.0
                km_str = f"AT Km: {km_val:.3f}"
                callout_lines = [
                    f"C/L OF {sm.station_name}",
                    "RAILWAY STATION,",
                    km_str,
                ]
                full_text = "\\P".join(callout_lines)

                msp.add_mtext(
                    full_text,
                    dxfattribs={
                        "layer": lay_text,
                        "char_height": th * 1.1,
                        "style": "ESP_STYLE",
                        "attachment_point": 1,  # TOP_LEFT
                        "insert": (p_top[0] + 2.0 * anno_scale, p_top[1]),
                    },
                )

        logger.info("Rendered yard '%s' onto DXF modelspace.", yard_tracks.station_name)

    def create_standalone_esp_dxf(
        self,
        yard_config: YardConfig,
        output_dxf_path: str | Path,
        sheet_size: str = "A1",
    ) -> ezdxf.document.Drawing:
        """Generate a complete standalone Indian Railways Engineering Scale Plan (ESP) DXF file."""
        doc = ezdxf.new(dxfversion="R2013")
        doc.units = units.M
        msp = doc.modelspace()
        self.setup_layers(doc)

        # Build straight reference alignment around station chainage
        st_ch = yard_config.centerline_chainage_m
        align_coords = [
            (st_ch - 1000.0, 0.0),
            (st_ch + 1000.0, 0.0),
        ]
        projector = AlignmentProjector(align_coords, start_chainage_m=st_ch - 1000.0)

        # Project tracks and civil elements
        yard_tracks = generate_projected_tracks(yard_config, projector)
        civil_layout = generate_projected_civil_elements(yard_config, projector)

        # Render
        self.render_yard(msp, yard_tracks, civil_layout)

        # Save
        out_p = Path(output_dxf_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        doc.saveas(out_p)
        logger.info("Saved standalone ESP drawing to: %s", out_p)
        return doc
