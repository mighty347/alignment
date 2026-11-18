"""Alignment Projector and Moving Frame (Frenet-Serret) Engine.

Transforms station yard components from local (Station Chainage s, Lateral Offset d)
coordinates into global Real-World / UTM / Paper coordinates (X, Y).

Supports:
  - Native Python coordinate sequences (zero mandatory dependencies)
  - Shapely LineString geometry objects
  - Arbitrary curved and straight alignments (calculating continuous tangent & normal vectors)
"""
from __future__ import annotations

import logging
import math
from typing import Any, Callable, Optional, Sequence, Union

logger = logging.getLogger("ESP_Yard_Engine.AlignmentProjector")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class AlignmentProjector:
    """Projects local (s, offset) coordinates onto a 2D reference railway alignment."""

    def __init__(
        self,
        reference_alignment: Any,
        start_chainage_m: float = 0.0,
        scale_factor: float = 1.0,
    ) -> None:
        """Initialize the projector with a reference alignment.

        Args:
            reference_alignment: Either:
              1. A Shapely ``LineString``
              2. A sequence of (x, y) or (x, y, z) tuples: ``[(x0, y0), (x1, y1), ...]``
              3. A callable function ``fn(chainage_m) -> (x, y, heading_rad)``
            start_chainage_m: The absolute chainage corresponding to the start of the alignment.
            scale_factor: Plan view scale factor (ground meters -> drawing units).
                          1.0 for ground meters; (draw_len_mm / draw_len_m) for sheet views.
        """
        self.start_chainage_m = float(start_chainage_m)
        self.scale_factor = float(scale_factor)
        self._shapely_line: Optional[Any] = None
        self._coords: list[tuple[float, float]] = []
        self._cumulative_lengths: list[float] = []
        self._callable_eval: Optional[Callable[[float], tuple[float, float, float]]] = None

        if callable(reference_alignment):
            self._callable_eval = reference_alignment
            self.total_length_m = 1e9  # Effectively unbounded for analytical callable
            logger.debug("Initialized AlignmentProjector with callable evaluation function.")
            return

        # Check for Shapely LineString
        type_name = type(reference_alignment).__name__
        if "LineString" in type_name or hasattr(reference_alignment, "interpolate"):
            self._shapely_line = reference_alignment
            self.total_length_m = float(reference_alignment.length)
            # Extract coordinates for fast lookup if needed
            self._coords = [(float(p[0]), float(p[1])) for p in reference_alignment.coords]
            self._build_cumulative_lengths()
            logger.debug("Initialized AlignmentProjector with Shapely LineString (Length: %.2fm).", self.total_length_m)
            return

        # Plain coordinate sequence
        if isinstance(reference_alignment, (list, tuple)):
            self._coords = [(float(p[0]), float(p[1])) for p in reference_alignment]
            self._build_cumulative_lengths()
            self.total_length_m = self._cumulative_lengths[-1] if self._cumulative_lengths else 0.0
            logger.debug("Initialized AlignmentProjector with %d coordinates (Length: %.2fm).", len(self._coords), self.total_length_m)
            return

        raise TypeError(
            f"Unsupported reference alignment type: {type(reference_alignment)}. "
            "Expected Shapely LineString, coordinate sequence, or callable function."
        )

    def _build_cumulative_lengths(self) -> None:
        """Compute cumulative Euclidean segment distances."""
        if not self._coords:
            self._cumulative_lengths = [0.0]
            return

        self._cumulative_lengths = [0.0]
        accum = 0.0
        for i in range(len(self._coords) - 1):
            dx = self._coords[i + 1][0] - self._coords[i][0]
            dy = self._coords[i + 1][1] - self._coords[i][1]
            seg_len = math.hypot(dx, dy)
            accum += seg_len
            self._cumulative_lengths.append(accum)

    def get_frame(
        self,
        chainage_m: float,
        delta_step_m: float = 0.5,
    ) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float], float]:
        """Compute the Frenet-Serret frame at the specified absolute chainage.

        Returns:
            (position (x, y), unit_tangent (tx, ty), unit_normal (nx, ny), heading_rad)
            - Unit normal points 90 degrees to the LEFT of the forward tangent vector.
        """
        if self._callable_eval is not None:
            px, py, heading = self._callable_eval(chainage_m)
            tx = math.cos(heading)
            ty = math.sin(heading)
            nx = -ty
            ny = tx
            return (px, py), (tx, ty), (nx, ny), heading

        local_s = chainage_m - self.start_chainage_m

        # Clamp local_s within bounds
        clamped_s = max(0.0, min(self.total_length_m, local_s))

        # Position evaluation
        if self._shapely_line is not None:
            pt = self._shapely_line.interpolate(clamped_s)
            px, py = float(pt.x), float(pt.y)

            # Central finite differences for tangent
            s_back = max(0.0, clamped_s - delta_step_m)
            s_fwd = min(self.total_length_m, clamped_s + delta_step_m)
            if s_fwd == s_back:
                s_fwd = s_back + 0.001

            pt_back = self._shapely_line.interpolate(s_back)
            pt_fwd = self._shapely_line.interpolate(s_fwd)
            dx = float(pt_fwd.x - pt_back.x)
            dy = float(pt_fwd.y - pt_back.y)
        else:
            # Native coordinate lookup with binary/linear search
            px, py, dx, dy = self._eval_native_coords(clamped_s, delta_step_m)

        mag = math.hypot(dx, dy)
        if mag < 1e-7:
            tx, ty = 1.0, 0.0
            heading = 0.0
        else:
            tx = dx / mag
            ty = dy / mag
            heading = math.atan2(ty, tx)

        # Unit Normal pointing 90 degrees LEFT (counter-clockwise)
        nx = -ty
        ny = tx

        return (px, py), (tx, ty), (nx, ny), heading

    def _eval_native_coords(
        self,
        local_s: float,
        delta_step_m: float,
    ) -> tuple[float, float, float, float]:
        """Interpolate point and forward derivative using cumulative length arrays."""
        if not self._coords:
            return 0.0, 0.0, 1.0, 0.0

        if len(self._coords) == 1:
            return self._coords[0][0], self._coords[0][1], 1.0, 0.0

        import bisect
        idx = bisect.bisect_right(self._cumulative_lengths, local_s)
        if idx <= 0:
            p0 = self._coords[0]
            p1 = self._coords[1]
            dx = p1[0] - p0[0]
            dy = p1[1] - p0[1]
            return p0[0], p0[1], dx, dy
        if idx >= len(self._cumulative_lengths):
            p0 = self._coords[-2]
            p1 = self._coords[-1]
            dx = p1[0] - p0[0]
            dy = p1[1] - p0[1]
            return p1[0], p1[1], dx, dy

        s0 = self._cumulative_lengths[idx - 1]
        s1 = self._cumulative_lengths[idx]
        seg_len = max(0.0001, s1 - s0)
        t = (local_s - s0) / seg_len

        p0 = self._coords[idx - 1]
        p1 = self._coords[idx]

        px = p0[0] + t * (p1[0] - p0[0])
        py = p0[1] + t * (p1[1] - p0[1])

        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        return px, py, dx, dy

    def project_point(
        self,
        chainage_m: float,
        lateral_offset_m: float,
    ) -> tuple[float, float]:
        """Project a single (chainage, offset) point into global (X, Y) coordinates.

        Formula:
          P_world = P_centerline(chainage) + (offset * scale_factor) * Normal_vector(chainage)
        """
        pos, _, norm, _ = self.get_frame(chainage_m)
        scaled_offset = lateral_offset_m * self.scale_factor
        gx = pos[0] + scaled_offset * norm[0]
        gy = pos[1] + scaled_offset * norm[1]
        return (gx, gy)

    def project_polyline(
        self,
        points_s_offset: Sequence[tuple[float, float]],
        base_chainage_m: float = 0.0,
    ) -> list[tuple[float, float]]:
        """Project a sequence of (relative_s, lateral_offset) points into global (X, Y) coordinates.

        Args:
            points_s_offset: List of (s_rel, offset) tuples.
            base_chainage_m: Base chainage to add to s_rel (typically Station Centerline chainage).

        Returns:
            List of global (X, Y) coordinates.
        """
        result: list[tuple[float, float]] = []
        for s_rel, offset in points_s_offset:
            abs_ch = base_chainage_m + s_rel
            gx, gy = self.project_point(abs_ch, offset)
            result.append((gx, gy))
        return result
