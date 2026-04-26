"""Cluster-and-anchor macro placer.

This submission uses a compact version of the idea from your pasted flow:
1. Group movable hard macros into connectivity-based clusters.
2. Assign each cluster a tethered anchor near the center of the canvas.
3. Place macros inside each cluster with legal ring-search around that anchor.
4. Keep fixed macros and soft macros at their original locations.

The evaluator only needs a class with a place(self, benchmark) method that
returns a torch.Tensor of shape [num_macros, 2].
"""

from __future__ import annotations

import math
from collections import deque

import torch

from macro_place.benchmark import Benchmark


class ClusterAnchorPlacer:
    """Connectivity-aware placer with cluster anchors and legal fallback search."""

    def __init__(
        self,
        cluster_size: int = 8,
        max_rings: int = 60,
        min_search_step: float = 2.0,
        fallback_grid_steps: int = 28,
        edge_gap: float = 0.001,
    ):
        self.cluster_size = max(1, int(cluster_size))
        self.max_rings = max(1, int(max_rings))
        self.min_search_step = float(min_search_step)
        self.fallback_grid_steps = max(4, int(fallback_grid_steps))
        self.edge_gap = float(edge_gap)

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        placement = benchmark.macro_positions.clone()

        hard_mask = benchmark.get_hard_macro_mask()
        movable_mask = benchmark.get_movable_mask() & hard_mask
        movable_indices = torch.where(movable_mask)[0].tolist()
        if not movable_indices:
            return placement

        num_hard = benchmark.num_hard_macros
        canvas_w = float(benchmark.canvas_width)
        canvas_h = float(benchmark.canvas_height)
        sizes = benchmark.macro_sizes
        degrees = self._macro_degrees(benchmark)

        # Build connectivity-based clusters, then split large components into chunks.
        clusters = self._build_clusters(benchmark, movable_mask, degrees)
        clusters.sort(
            key=lambda cluster: (
                -sum(float(degrees[idx].item()) for idx in cluster),
                -sum(float((sizes[idx, 0] * sizes[idx, 1]).item()) for idx in cluster),
                -len(cluster),
            )
        )

        placed_hard = torch.zeros(num_hard, dtype=torch.bool)
        placed_hard[~movable_mask[:num_hard]] = True

        # Sort movable macros globally by size (largest first) to prevent boxing
        all_movable_sorted = sorted(
            movable_indices,
            key=lambda idx: (
                -float((sizes[idx, 0] * sizes[idx, 1]).item()),
                -float(sizes[idx, 1].item()),
                -float(sizes[idx, 0].item()),
            ),
        )

        # Place all macros in size order, using clusters only for anchoring
        cluster_assignments = {}
        for cluster_id, cluster in enumerate(clusters):
            for idx in cluster:
                cluster_assignments[idx] = cluster_id

        anchors = self._cluster_anchors(len(clusters), canvas_w, canvas_h)

        for rank, idx in enumerate(all_movable_sorted):
            cluster_id = cluster_assignments.get(idx, 0)
            anchor_x, anchor_y = anchors[cluster_id]
            # Compute cluster radius based on all macros in cluster
            cluster = clusters[cluster_id]
            cluster_radius = self._cluster_radius(cluster, sizes)

            target_x, target_y = self._macro_target(
                idx=idx,
                rank=rank,
                total=len(all_movable_sorted),
                anchor_x=anchor_x,
                anchor_y=anchor_y,
                cluster_radius=cluster_radius,
                sizes=sizes,
                canvas_w=canvas_w,
                canvas_h=canvas_h,
            )
            x, y = self._find_legal_position(
                idx=idx,
                target_x=target_x,
                target_y=target_y,
                placement=placement,
                placed_hard=placed_hard,
                sizes=sizes,
                canvas_w=canvas_w,
                canvas_h=canvas_h,
            )
            placement[idx, 0] = x
            placement[idx, 1] = y
            placed_hard[idx] = True

        # Final aggressive legalization pass
        self._final_legalize_overlaps(placement, all_movable_sorted, placed_hard, sizes, canvas_w, canvas_h)

        return placement

    def _macro_degrees(self, benchmark: Benchmark) -> torch.Tensor:
        degrees = torch.zeros(benchmark.num_hard_macros, dtype=torch.float32)
        for net in benchmark.net_nodes:
            hard_nodes = net[net < benchmark.num_hard_macros]
            if hard_nodes.numel() < 2:
                continue
            unique_nodes = torch.unique(hard_nodes)
            contribution = float(unique_nodes.numel() - 1)
            for idx in unique_nodes.tolist():
                degrees[idx] += contribution
        return degrees

    def _build_clusters(
        self,
        benchmark: Benchmark,
        movable_mask: torch.Tensor,
        degrees: torch.Tensor,
    ) -> list[list[int]]:
        num_hard = benchmark.num_hard_macros
        parent = list(range(num_hard))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra = find(a)
            rb = find(b)
            if ra != rb:
                parent[rb] = ra

        # Connectivity-based grouping on the hard-macro projection of each net.
        for net in benchmark.net_nodes:
            hard_nodes = [int(idx) for idx in net.tolist() if idx < num_hard and movable_mask[idx]]
            if len(hard_nodes) < 2:
                continue
            first = hard_nodes[0]
            for other in hard_nodes[1:]:
                union(first, other)

        components: dict[int, list[int]] = {}
        for idx in torch.where(movable_mask[:num_hard])[0].tolist():
            root = find(idx)
            components.setdefault(root, []).append(idx)

        clusters: list[list[int]] = []
        for component in components.values():
            component.sort(
                key=lambda idx: (
                    -float(degrees[idx].item()),
                    -float((benchmark.macro_sizes[idx, 0] * benchmark.macro_sizes[idx, 1]).item()),
                )
            )
            for start in range(0, len(component), self.cluster_size):
                clusters.append(component[start : start + self.cluster_size])

        return clusters

    def _cluster_anchors(self, count: int, canvas_w: float, canvas_h: float) -> list[tuple[float, float]]:
        if count <= 0:
            return []

        center_x = canvas_w * 0.5
        center_y = canvas_h * 0.5
        max_rx = canvas_w * 0.32
        max_ry = canvas_h * 0.32
        golden_angle = 2.399963229728653

        anchors: list[tuple[float, float]] = []
        for i in range(count):
            if count == 1:
                radial = 0.0
            else:
                radial = (i + 0.5) / float(count)
            theta = i * golden_angle
            x = center_x + math.cos(theta) * max_rx * radial
            y = center_y + math.sin(theta) * max_ry * radial
            anchors.append((x, y))
        return anchors

    def _cluster_radius(self, cluster: list[int], sizes: torch.Tensor) -> float:
        if not cluster:
            return self.min_search_step
        scale = 0.0
        for idx in cluster:
            scale += float(max(sizes[idx, 0].item(), sizes[idx, 1].item()))
        return max(self.min_search_step, scale / max(1, len(cluster)) * 0.6)

    def _macro_target(
        self,
        idx: int,
        rank: int,
        total: int,
        anchor_x: float,
        anchor_y: float,
        cluster_radius: float,
        sizes: torch.Tensor,
        canvas_w: float,
        canvas_h: float,
    ) -> tuple[float, float]:
        w = float(sizes[idx, 0].item())
        h = float(sizes[idx, 1].item())

        if total <= 1:
            radial = 0.0
        else:
            radial = rank / float(total - 1)

        theta = (idx + 1) * 2.399963229728653
        offset_scale = 0.25 + 0.75 * radial
        x = anchor_x + math.cos(theta) * cluster_radius * offset_scale
        y = anchor_y + math.sin(theta) * cluster_radius * offset_scale

        x = min(max(x, w * 0.5), canvas_w - w * 0.5)
        y = min(max(y, h * 0.5), canvas_h - h * 0.5)
        return x, y

    def _find_legal_position(
        self,
        idx: int,
        target_x: float,
        target_y: float,
        placement: torch.Tensor,
        placed_hard: torch.Tensor,
        sizes: torch.Tensor,
        canvas_w: float,
        canvas_h: float,
    ) -> tuple[float, float]:
        for margin_scale in (1.0, 0.5, 0.0):
            candidate = self._ring_search(
                idx,
                target_x,
                target_y,
                placement,
                placed_hard,
                sizes,
                canvas_w,
                canvas_h,
                margin_scale,
            )
            if candidate is not None:
                return candidate

        candidate = self._grid_fallback(
            idx,
            target_x,
            target_y,
            placement,
            placed_hard,
            sizes,
            canvas_w,
            canvas_h,
        )
        if candidate is not None:
            return candidate

        w = float(sizes[idx, 0].item())
        h = float(sizes[idx, 1].item())
        return (
            min(max(target_x, w * 0.5), canvas_w - w * 0.5),
            min(max(target_y, h * 0.5), canvas_h - h * 0.5),
        )

    def _ring_search(
        self,
        idx: int,
        target_x: float,
        target_y: float,
        placement: torch.Tensor,
        placed_hard: torch.Tensor,
        sizes: torch.Tensor,
        canvas_w: float,
        canvas_h: float,
        margin_scale: float,
    ) -> tuple[float, float] | None:
        w = float(sizes[idx, 0].item())
        h = float(sizes[idx, 1].item())
        x_min = w * 0.5
        x_max = canvas_w - w * 0.5
        y_min = h * 0.5
        y_max = canvas_h - h * 0.5

        base_x = min(max(target_x, x_min), x_max)
        base_y = min(max(target_y, y_min), y_max)

        if self._is_legal(idx, base_x, base_y, placement, placed_hard, sizes, margin_scale):
            return base_x, base_y

        step_x = max(self.min_search_step, w * 0.35)
        step_y = max(self.min_search_step, h * 0.35)

        for ring in range(1, self.max_rings + 1):
            radius_x = ring * step_x
            radius_y = ring * step_y
            samples = 8 + ring * 4
            best = None
            best_score = float("inf")

            for sample in range(samples):
                theta = (2.0 * math.pi * sample) / samples
                x = min(max(base_x + radius_x * math.cos(theta), x_min), x_max)
                y = min(max(base_y + radius_y * math.sin(theta), y_min), y_max)

                if not self._is_legal(idx, x, y, placement, placed_hard, sizes, margin_scale):
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
        canvas_w: float,
        canvas_h: float,
    ) -> tuple[float, float] | None:
        w = float(sizes[idx, 0].item())
        h = float(sizes[idx, 1].item())
        x_min = w * 0.5
        x_max = canvas_w - w * 0.5
        y_min = h * 0.5
        y_max = canvas_h - h * 0.5

        xs = torch.linspace(x_min, x_max, steps=self.fallback_grid_steps).tolist()
        ys = torch.linspace(y_min, y_max, steps=self.fallback_grid_steps).tolist()
        xs.sort(key=lambda value: abs(value - target_x))
        ys.sort(key=lambda value: abs(value - target_y))

        for y in ys:
            for x in xs:
                if self._is_legal(idx, x, y, placement, placed_hard, sizes, 0.0):
                    return x, y
        return None

    def _is_legal(
        self,
        idx: int,
        x: float,
        y: float,
        placement: torch.Tensor,
        placed_hard: torch.Tensor,
        sizes: torch.Tensor,
        margin_scale: float,
    ) -> bool:
        w_i = float(sizes[idx, 0].item())
        h_i = float(sizes[idx, 1].item())

        for j in torch.where(placed_hard)[0].tolist():
            if j == idx:
                continue
            x_j = float(placement[j, 0].item())
            y_j = float(placement[j, 1].item())
            w_j = float(sizes[j, 0].item())
            h_j = float(sizes[j, 1].item())

            extra_margin = margin_scale * 0.15 * max(w_i, h_i, w_j, h_j)
            sep_x = (w_i + w_j) * 0.5 + self.edge_gap + extra_margin
            sep_y = (h_i + h_j) * 0.5 + self.edge_gap + extra_margin
            if abs(x - x_j) < sep_x and abs(y - y_j) < sep_y:
                return False

        return True

    def _final_legalize_overlaps(
        self,
        placement: torch.Tensor,
        movable_indices: list[int],
        placed_hard: torch.Tensor,
        sizes: torch.Tensor,
        canvas_w: float,
        canvas_h: float,
    ) -> None:
        """Quick final pass targeting only macros with actual overlaps."""
        for _ in range(2):
            overlapping = []
            for idx in movable_indices:
                x = float(placement[idx, 0].item())
                y = float(placement[idx, 1].item())
                if not self._is_legal(idx, x, y, placement, placed_hard, sizes, 0.0):
                    overlapping.append(idx)
            
            if not overlapping:
                break
            
            for idx in overlapping:
                target_x = float(placement[idx, 0].item())
                target_y = float(placement[idx, 1].item())
                candidate = self._ring_search(
                    idx, target_x, target_y, placement, placed_hard, sizes, canvas_w, canvas_h, 0.0
                )
                if candidate is None:
                    candidate = self._grid_fallback(
                        idx, target_x, target_y, placement, placed_hard, sizes, canvas_w, canvas_h
                    )
                if candidate is not None:
                    placement[idx, 0] = candidate[0]
                    placement[idx, 1] = candidate[1]
