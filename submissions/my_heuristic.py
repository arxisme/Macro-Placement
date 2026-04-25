"""
Connectivity-Aware Center-Biased Heuristic Placer.

Key ideas:
1. Sort movable hard macros by connectivity degree (highest first)
2. Assign high-degree macros center-biased targets
3. Legalize with overlap-safe ring search and adaptive keep-out relaxation

Usage:
    uv run evaluate submissions/my_heuristic.py -b ibm01
    uv run evaluate submissions/my_heuristic.py --all
"""

import math
import torch

from macro_place.benchmark import Benchmark


class ConnectivityCenterPlacer:
    """Connectivity-aware placer with center-biased target generation."""

    def __init__(
        self,
        base_keepout_ratio: float = 0.06,
        edge_gap: float = 0.001,
        min_search_step: float = 2.0,
        max_rings: int = 90,
        fallback_grid_steps: int = 36,
        legalization_passes: int = 2,
    ):
        self.base_keepout_ratio = base_keepout_ratio
        self.edge_gap = edge_gap
        self.min_search_step = min_search_step
        self.max_rings = max_rings
        self.fallback_grid_steps = fallback_grid_steps
        self.legalization_passes = legalization_passes

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        placement = benchmark.macro_positions.clone()

        hard_mask = benchmark.get_hard_macro_mask()
        movable = benchmark.get_movable_mask() & hard_mask
        movable_indices = torch.where(movable)[0].tolist()

        if not movable_indices:
            return placement

        sizes = benchmark.macro_sizes
        canvas_w = float(benchmark.canvas_width)
        canvas_h = float(benchmark.canvas_height)
        center_x = canvas_w * 0.5
        center_y = canvas_h * 0.5

        degrees = self._compute_macro_degrees(benchmark)

        # Higher connectivity first; tie-break with larger area.
        movable_indices.sort(
            key=lambda i: (
                -float(degrees[i].item()),
                -float((sizes[i, 0] * sizes[i, 1]).item()),
                -float(sizes[i, 1].item()),
            )
        )

        num_hard = benchmark.num_hard_macros
        placed_hard = torch.zeros(num_hard, dtype=torch.bool)

        # Fixed hard macros are immutable obstacles.
        for i in range(num_hard):
            if not movable[i]:
                placed_hard[i] = True

        keepout = self._build_keepout(sizes[:num_hard], degrees)

        total = len(movable_indices)
        for rank, idx in enumerate(movable_indices):
            target_x, target_y = self._target_from_rank(
                idx,
                rank,
                total,
                sizes,
                center_x,
                center_y,
                canvas_w,
                canvas_h,
            )

            x, y = self._find_legal_position(
                idx=idx,
                target_x=target_x,
                target_y=target_y,
                placement=placement,
                placed_hard=placed_hard,
                sizes=sizes,
                keepout=keepout,
                canvas_w=canvas_w,
                canvas_h=canvas_h,
            )

            placement[idx, 0] = x
            placement[idx, 1] = y
            placed_hard[idx] = True

        self._legalize_overlaps(
            placement=placement,
            movable_indices=movable_indices,
            sizes=sizes,
            keepout=keepout,
            canvas_w=canvas_w,
            canvas_h=canvas_h,
        )

        return placement

    def _compute_macro_degrees(self, benchmark: Benchmark) -> torch.Tensor:
        """
        Return connectivity degree per hard macro.

        Preferred path uses benchmark.netlist.get_macro_degree(i) if present.
        Fallback computes hyperedge degree from benchmark.net_nodes.
        """
        num_hard = benchmark.num_hard_macros
        degrees = torch.zeros(num_hard, dtype=torch.float32)

        netlist = getattr(benchmark, "netlist", None)
        if netlist is not None and hasattr(netlist, "get_macro_degree"):
            for i in range(num_hard):
                degrees[i] = float(netlist.get_macro_degree(i))
            return degrees

        for net_nodes in benchmark.net_nodes:
            hard_nodes = net_nodes[net_nodes < num_hard]
            if hard_nodes.numel() <= 1:
                continue
            unique_hard = torch.unique(hard_nodes)
            contribution = float(unique_hard.numel() - 1)
            for idx in unique_hard.tolist():
                degrees[idx] += contribution

        return degrees

    def _build_keepout(self, hard_sizes: torch.Tensor, degrees: torch.Tensor) -> torch.Tensor:
        size_scale = torch.maximum(hard_sizes[:, 0], hard_sizes[:, 1])
        max_degree = max(float(torch.max(degrees).item()), 1.0)
        degree_scale = 0.5 + (degrees / max_degree)
        return self.base_keepout_ratio * size_scale * degree_scale

    def _target_from_rank(
        self,
        idx: int,
        rank: int,
        total: int,
        sizes: torch.Tensor,
        center_x: float,
        center_y: float,
        canvas_w: float,
        canvas_h: float,
    ) -> tuple[float, float]:
        if total <= 1:
            radial_fraction = 0.0
        else:
            radial_fraction = rank / float(total - 1)

        w = float(sizes[idx, 0].item())
        h = float(sizes[idx, 1].item())

        max_rx = max(0.0, canvas_w * 0.48 - w * 0.5)
        max_ry = max(0.0, canvas_h * 0.48 - h * 0.5)

        radius_x = radial_fraction * max_rx
        radius_y = radial_fraction * max_ry

        # Deterministic angular spread to avoid clustering on one axis.
        golden_angle = 2.399963229728653
        theta = (idx * golden_angle) % (2.0 * math.pi)

        target_x = center_x + radius_x * math.cos(theta)
        target_y = center_y + radius_y * math.sin(theta)

        x_min = w * 0.5
        x_max = canvas_w - w * 0.5
        y_min = h * 0.5
        y_max = canvas_h - h * 0.5

        target_x = min(max(target_x, x_min), x_max)
        target_y = min(max(target_y, y_min), y_max)
        return target_x, target_y

    def _find_legal_position(
        self,
        idx: int,
        target_x: float,
        target_y: float,
        placement: torch.Tensor,
        placed_hard: torch.Tensor,
        sizes: torch.Tensor,
        keepout: torch.Tensor,
        canvas_w: float,
        canvas_h: float,
    ) -> tuple[float, float]:
        # Adaptive relaxation: try stronger keep-out first, then relax to pure legality.
        for margin_scale in (1.0, 0.5, 0.0):
            candidate = self._ring_search(
                idx,
                target_x,
                target_y,
                placement,
                placed_hard,
                sizes,
                keepout,
                margin_scale,
                canvas_w,
                canvas_h,
            )
            if candidate is not None:
                return candidate

        # Dense fallback over the canvas to preserve legality in hard cases.
        fallback = self._grid_fallback(
            idx,
            target_x,
            target_y,
            placement,
            placed_hard,
            sizes,
            keepout,
            canvas_w,
            canvas_h,
        )
        if fallback is not None:
            return fallback

        raster = self._raster_fallback(
            idx,
            target_x,
            target_y,
            placement,
            placed_hard,
            sizes,
            keepout,
            canvas_w,
            canvas_h,
        )
        if raster is not None:
            return raster

        # Last resort: clipped target (may be invalid if impossible to legalize).
        w = float(sizes[idx, 0].item())
        h = float(sizes[idx, 1].item())
        x = min(max(target_x, w * 0.5), canvas_w - w * 0.5)
        y = min(max(target_y, h * 0.5), canvas_h - h * 0.5)
        return x, y

    def _ring_search(
        self,
        idx: int,
        target_x: float,
        target_y: float,
        placement: torch.Tensor,
        placed_hard: torch.Tensor,
        sizes: torch.Tensor,
        keepout: torch.Tensor,
        margin_scale: float,
        canvas_w: float,
        canvas_h: float,
    ) -> tuple[float, float] | None:
        w = float(sizes[idx, 0].item())
        h = float(sizes[idx, 1].item())

        x_min = w * 0.5
        x_max = canvas_w - w * 0.5
        y_min = h * 0.5
        y_max = canvas_h - h * 0.5

        base_x = min(max(target_x, x_min), x_max)
        base_y = min(max(target_y, y_min), y_max)

        if self._is_legal(
            idx,
            base_x,
            base_y,
            placement,
            placed_hard,
            sizes,
            keepout,
            margin_scale,
        ):
            return base_x, base_y

        step_x = max(w * 0.35, self.min_search_step)
        step_y = max(h * 0.35, self.min_search_step)

        for ring in range(1, self.max_rings + 1):
            radius_x = ring * step_x
            radius_y = ring * step_y
            samples = 12 + ring * 4

            best = None
            best_score = float("inf")
            for k in range(samples):
                theta = (2.0 * math.pi * k) / samples
                x = min(max(base_x + radius_x * math.cos(theta), x_min), x_max)
                y = min(max(base_y + radius_y * math.sin(theta), y_min), y_max)

                if not self._is_legal(
                    idx,
                    x,
                    y,
                    placement,
                    placed_hard,
                    sizes,
                    keepout,
                    margin_scale,
                ):
                    continue

                score = (x - target_x) ** 2 + (y - target_y) ** 2
                if score < best_score:
                    best_score = score
                    best = (x, y)

            if best is not None:
                return best

        return None

    def _grid_fallback(
        self,
        idx: int,
        target_x: float,
        target_y: float,
        placement: torch.Tensor,
        placed_hard: torch.Tensor,
        sizes: torch.Tensor,
        keepout: torch.Tensor,
        canvas_w: float,
        canvas_h: float,
    ) -> tuple[float, float] | None:
        w = float(sizes[idx, 0].item())
        h = float(sizes[idx, 1].item())
        x_min = w * 0.5
        x_max = canvas_w - w * 0.5
        y_min = h * 0.5
        y_max = canvas_h - h * 0.5

        xs = torch.linspace(x_min, x_max, steps=self.fallback_grid_steps)
        ys = torch.linspace(y_min, y_max, steps=self.fallback_grid_steps)

        best = None
        best_score = float("inf")

        for y in ys.tolist():
            for x in xs.tolist():
                if not self._is_legal(
                    idx,
                    x,
                    y,
                    placement,
                    placed_hard,
                    sizes,
                    keepout,
                    margin_scale=0.0,
                ):
                    continue

                score = (x - target_x) ** 2 + (y - target_y) ** 2
                if score < best_score:
                    best_score = score
                    best = (x, y)

        return best

    def _raster_fallback(
        self,
        idx: int,
        target_x: float,
        target_y: float,
        placement: torch.Tensor,
        placed_hard: torch.Tensor,
        sizes: torch.Tensor,
        keepout: torch.Tensor,
        canvas_w: float,
        canvas_h: float,
    ) -> tuple[float, float] | None:
        w = float(sizes[idx, 0].item())
        h = float(sizes[idx, 1].item())

        x_min = w * 0.5
        x_max = canvas_w - w * 0.5
        y_min = h * 0.5
        y_max = canvas_h - h * 0.5

        x_step = max(0.5, min(w * 0.2, 4.0))
        y_step = max(0.5, min(h * 0.2, 4.0))

        x_count = max(2, int((x_max - x_min) / x_step) + 1)
        y_count = max(2, int((y_max - y_min) / y_step) + 1)

        xs = torch.linspace(x_min, x_max, steps=x_count).tolist()
        ys = torch.linspace(y_min, y_max, steps=y_count).tolist()

        xs.sort(key=lambda val: abs(val - target_x))
        ys.sort(key=lambda val: abs(val - target_y))

        for y in ys:
            for x in xs:
                if self._is_legal(
                    idx,
                    x,
                    y,
                    placement,
                    placed_hard,
                    sizes,
                    keepout,
                    margin_scale=0.0,
                ):
                    return x, y

        return None

    def _legalize_overlaps(
        self,
        placement: torch.Tensor,
        movable_indices: list[int],
        sizes: torch.Tensor,
        keepout: torch.Tensor,
        canvas_w: float,
        canvas_h: float,
    ):
        num_hard = keepout.shape[0]
        blockers = torch.ones(num_hard, dtype=torch.bool)

        for _ in range(self.legalization_passes):
            if not self._has_any_hard_overlap(placement, sizes, num_hard):
                return

            moved = False
            # Move lower-degree (outer-ring) macros first to preserve center intent.
            for idx in reversed(movable_indices):
                blockers[idx] = False

                x0 = float(placement[idx, 0].item())
                y0 = float(placement[idx, 1].item())
                already_legal = self._is_legal(
                    idx,
                    x0,
                    y0,
                    placement,
                    blockers,
                    sizes,
                    keepout,
                    margin_scale=0.0,
                )
                if already_legal:
                    blockers[idx] = True
                    continue

                candidate = self._ring_search(
                    idx,
                    x0,
                    y0,
                    placement,
                    blockers,
                    sizes,
                    keepout,
                    margin_scale=0.0,
                    canvas_w=canvas_w,
                    canvas_h=canvas_h,
                )

                if candidate is None:
                    candidate = self._grid_fallback(
                        idx,
                        x0,
                        y0,
                        placement,
                        blockers,
                        sizes,
                        keepout,
                        canvas_w,
                        canvas_h,
                    )

                if candidate is None:
                    candidate = self._raster_fallback(
                        idx,
                        x0,
                        y0,
                        placement,
                        blockers,
                        sizes,
                        keepout,
                        canvas_w,
                        canvas_h,
                    )

                if candidate is not None:
                    placement[idx, 0] = candidate[0]
                    placement[idx, 1] = candidate[1]
                    moved = True

                blockers[idx] = True

            if not moved:
                return

    def _has_any_hard_overlap(
        self,
        placement: torch.Tensor,
        sizes: torch.Tensor,
        num_hard: int,
    ) -> bool:
        for i in range(num_hard):
            x_i = float(placement[i, 0].item())
            y_i = float(placement[i, 1].item())
            w_i = float(sizes[i, 0].item())
            h_i = float(sizes[i, 1].item())
            for j in range(i + 1, num_hard):
                x_j = float(placement[j, 0].item())
                y_j = float(placement[j, 1].item())
                w_j = float(sizes[j, 0].item())
                h_j = float(sizes[j, 1].item())
                if (
                    abs(x_i - x_j) < (w_i + w_j) * 0.5 + self.edge_gap
                    and abs(y_i - y_j) < (h_i + h_j) * 0.5 + self.edge_gap
                ):
                    return True
        return False

    def _is_legal(
        self,
        idx: int,
        x: float,
        y: float,
        placement: torch.Tensor,
        placed_hard: torch.Tensor,
        sizes: torch.Tensor,
        keepout: torch.Tensor,
        margin_scale: float,
    ) -> bool:
        w_i = float(sizes[idx, 0].item())
        h_i = float(sizes[idx, 1].item())
        keep_i = float(keepout[idx].item())

        blocked = torch.where(placed_hard)[0].tolist()
        for j in blocked:
            if j == idx:
                continue

            x_j = float(placement[j, 0].item())
            y_j = float(placement[j, 1].item())
            w_j = float(sizes[j, 0].item())
            h_j = float(sizes[j, 1].item())
            keep_j = float(keepout[j].item())

            margin = margin_scale * max(keep_i, keep_j)
            sep_x = (w_i + w_j) * 0.5 + self.edge_gap + margin
            sep_y = (h_i + h_j) * 0.5 + self.edge_gap + margin

            if abs(x - x_j) < sep_x and abs(y - y_j) < sep_y:
                return False

        return True