"""Cluster-and-anchor macro placer — fast, self-contained.

Everything is in this one file so the evaluator can find the placer class.

Usage:
    uv run evaluate submissions/cluster_anchor_placer.py -b ibm01
    uv run evaluate submissions/cluster_anchor_placer.py --all
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from macro_place.benchmark import Benchmark


# ---------------------------------------------------------------------------
# Spatial grid — fast overlap queries
# ---------------------------------------------------------------------------

class SpatialGrid:
    """
    Divides the canvas into cells of size `cell_size`.
    Overlap queries only check the 3x3 neighbourhood → 10-50x faster
    than checking all placed macros every time.
    """

    def __init__(self, canvas_w: float, canvas_h: float, cell_size: float):
        self.cell_size = max(cell_size, 1e-6)
        self.cols = max(1, int(math.ceil(canvas_w / self.cell_size)))
        self.rows = max(1, int(math.ceil(canvas_h / self.cell_size)))
        self.grid: list[list[list[int]]] = [
            [[] for _ in range(self.cols)] for _ in range(self.rows)
        ]

    def _cell(self, x: float, y: float) -> tuple[int, int]:
        col = min(max(int(x / self.cell_size), 0), self.cols - 1)
        row = min(max(int(y / self.cell_size), 0), self.rows - 1)
        return row, col

    def insert(self, idx: int, x: float, y: float) -> None:
        r, c = self._cell(x, y)
        self.grid[r][c].append(idx)

    def remove(self, idx: int, x: float, y: float) -> None:
        r, c = self._cell(x, y)
        try:
            self.grid[r][c].remove(idx)
        except ValueError:
            pass

    def neighbours(self, x: float, y: float) -> list[int]:
        r, c = self._cell(x, y)
        result = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    result.extend(self.grid[nr][nc])
        return result


# ---------------------------------------------------------------------------
# Main placer class  ← evaluator discovers this
# ---------------------------------------------------------------------------

class ClusterAnchorPlacer:
    """
    Connectivity-aware greedy macro placer with spatial-grid legality checks.

    Algorithm
    ---------
    1. Union-Find clusters of macros connected by nets.
    2. Assign each cluster a golden-angle spiral anchor on the canvas.
    3. Place macros largest-first; for each macro, ring-search outward
       from its anchor target until a legal (non-overlapping) position
       is found.  Uses a SpatialGrid so legality checks are O(1) instead
       of O(n).
    4. Final legalization pass to fix any remaining overlaps.
    """

    def __init__(
        self,
        cluster_size: int        = 8,
        max_rings: int           = 60,
        min_search_step: float   = 2.0,
        fallback_grid_steps: int = 28,
        edge_gap: float          = 0.001,
    ):
        self.cluster_size        = max(1, int(cluster_size))
        self.max_rings           = max(1, int(max_rings))
        self.min_search_step     = float(min_search_step)
        self.fallback_grid_steps = max(4, int(fallback_grid_steps))
        self.edge_gap            = float(edge_gap)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        placement = benchmark.macro_positions.clone()

        hard_mask       = benchmark.get_hard_macro_mask()
        movable_mask    = benchmark.get_movable_mask() & hard_mask
        movable_indices = torch.where(movable_mask)[0].tolist()

        if not movable_indices:
            return placement

        num_hard = benchmark.num_hard_macros
        canvas_w = float(benchmark.canvas_width)
        canvas_h = float(benchmark.canvas_height)
        sizes    = benchmark.macro_sizes

        # Build spatial grid sized to the largest macro
        max_dim   = float(sizes[movable_indices].max().item())
        cell_size = max(max_dim, self.min_search_step)
        grid      = SpatialGrid(canvas_w, canvas_h, cell_size)

        # Pre-populate grid with fixed macros
        placed_hard = torch.zeros(num_hard, dtype=torch.bool)
        placed_hard[~movable_mask[:num_hard]] = True
        for idx in torch.where(placed_hard)[0].tolist():
            grid.insert(idx,
                        float(placement[idx, 0].item()),
                        float(placement[idx, 1].item()))

        # Build clusters and assign anchors
        degrees  = self._macro_degrees(benchmark)
        clusters = self._build_clusters(benchmark, movable_mask, degrees)
        clusters.sort(key=lambda cl: (
            -sum(float(degrees[i].item()) for i in cl),
            -sum(float((sizes[i,0]*sizes[i,1]).item()) for i in cl),
            -len(cl),
        ))

        cluster_of = {idx: cid for cid, cl in enumerate(clusters) for idx in cl}
        anchors    = self._cluster_anchors(len(clusters), canvas_w, canvas_h)

        # Sort movable macros largest-first
        all_sorted = sorted(
            movable_indices,
            key=lambda i: (
                -float((sizes[i,0]*sizes[i,1]).item()),
                -float(sizes[i,1].item()),
                -float(sizes[i,0].item()),
            ),
        )

        # Place each macro
        for rank, idx in enumerate(all_sorted):
            cid    = cluster_of.get(idx, 0)
            ax, ay = anchors[cid]
            cl_r   = self._cluster_radius(clusters[cid], sizes)
            tx, ty = self._macro_target(
                idx, rank, len(all_sorted), ax, ay, cl_r,
                sizes, canvas_w, canvas_h,
            )
            x, y = self._find_legal(
                idx, tx, ty, placement, placed_hard,
                sizes, canvas_w, canvas_h, grid,
            )
            placement[idx, 0] = x
            placement[idx, 1] = y
            placed_hard[idx]  = True
            grid.insert(idx, x, y)

        # Final legalization
        self._legalize(
            placement, all_sorted, placed_hard,
            sizes, canvas_w, canvas_h, grid,
        )

        return placement

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------

    def _macro_degrees(self, benchmark: Benchmark) -> torch.Tensor:
        deg = torch.zeros(benchmark.num_hard_macros, dtype=torch.float32)
        for net in benchmark.net_nodes:
            hard = net[net < benchmark.num_hard_macros]
            if hard.numel() < 2:
                continue
            u = torch.unique(hard)
            c = float(u.numel() - 1)
            for i in u.tolist():
                deg[i] += c
        return deg

    def _build_clusters(
        self,
        benchmark: Benchmark,
        movable_mask: torch.Tensor,
        degrees: torch.Tensor,
    ) -> list[list[int]]:
        n      = benchmark.num_hard_macros
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for net in benchmark.net_nodes:
            nodes = [int(i) for i in net.tolist()
                     if i < n and movable_mask[i]]
            if len(nodes) >= 2:
                for other in nodes[1:]:
                    union(nodes[0], other)

        comps: dict[int, list[int]] = {}
        for idx in torch.where(movable_mask[:n])[0].tolist():
            comps.setdefault(find(idx), []).append(idx)

        sz = benchmark.macro_sizes
        clusters = []
        for comp in comps.values():
            comp.sort(key=lambda i: (
                -float(degrees[i].item()),
                -float((sz[i,0]*sz[i,1]).item()),
            ))
            for start in range(0, len(comp), self.cluster_size):
                clusters.append(comp[start:start + self.cluster_size])
        return clusters

    def _cluster_anchors(
        self, count: int, canvas_w: float, canvas_h: float
    ) -> list[tuple[float, float]]:
        cx, cy  = canvas_w * 0.5, canvas_h * 0.5
        rx, ry  = canvas_w * 0.32, canvas_h * 0.32
        golden  = 2.399963229728653
        anchors = []
        for i in range(max(1, count)):
            radial = (i + 0.5) / float(count) if count > 1 else 0.0
            theta  = i * golden
            anchors.append((
                cx + math.cos(theta) * rx * radial,
                cy + math.sin(theta) * ry * radial,
            ))
        return anchors

    def _cluster_radius(self, cluster: list[int], sizes: torch.Tensor) -> float:
        if not cluster:
            return self.min_search_step
        avg = sum(float(max(sizes[i,0].item(), sizes[i,1].item()))
                  for i in cluster) / len(cluster)
        return max(self.min_search_step, avg * 0.6)

    def _macro_target(
        self, idx, rank, total, ax, ay, cl_r, sizes, canvas_w, canvas_h
    ) -> tuple[float, float]:
        w = float(sizes[idx, 0].item())
        h = float(sizes[idx, 1].item())
        radial = rank / float(total - 1) if total > 1 else 0.0
        theta  = (idx + 1) * 2.399963229728653
        scale  = 0.25 + 0.75 * radial
        x = min(max(ax + math.cos(theta) * cl_r * scale, w*0.5), canvas_w - w*0.5)
        y = min(max(ay + math.sin(theta) * cl_r * scale, h*0.5), canvas_h - h*0.5)
        return x, y

    # ------------------------------------------------------------------
    # Legality (spatial-grid accelerated)
    # ------------------------------------------------------------------

    def _is_legal(
        self, idx: int, x: float, y: float,
        placement: torch.Tensor, placed_hard: torch.Tensor,
        sizes: torch.Tensor, grid: SpatialGrid,
        margin_scale: float = 0.0,
    ) -> bool:
        w_i = float(sizes[idx, 0].item())
        h_i = float(sizes[idx, 1].item())
        for j in grid.neighbours(x, y):
            if j == idx:
                continue
            x_j = float(placement[j, 0].item())
            y_j = float(placement[j, 1].item())
            w_j = float(sizes[j, 0].item())
            h_j = float(sizes[j, 1].item())
            extra = margin_scale * 0.15 * max(w_i, h_i, w_j, h_j)
            sep_x = (w_i + w_j) * 0.5 + self.edge_gap + extra
            sep_y = (h_i + h_j) * 0.5 + self.edge_gap + extra
            if abs(x - x_j) < sep_x and abs(y - y_j) < sep_y:
                return False
        return True

    def _find_legal(
        self, idx, tx, ty, placement, placed_hard,
        sizes, canvas_w, canvas_h, grid,
    ) -> tuple[float, float]:
        for margin in (1.0, 0.5, 0.0):
            pos = self._ring_search(
                idx, tx, ty, placement, placed_hard,
                sizes, canvas_w, canvas_h, grid, margin,
            )
            if pos is not None:
                return pos
        pos = self._grid_fallback(
            idx, tx, ty, placement, placed_hard,
            sizes, canvas_w, canvas_h, grid,
        )
        if pos is not None:
            return pos
        # Last resort: clamp to canvas, accept the overlap
        w = float(sizes[idx, 0].item())
        h = float(sizes[idx, 1].item())
        return (
            min(max(tx, w*0.5), canvas_w - w*0.5),
            min(max(ty, h*0.5), canvas_h - h*0.5),
        )

    def _ring_search(
        self, idx, tx, ty, placement, placed_hard,
        sizes, canvas_w, canvas_h, grid, margin_scale,
    ) -> Optional[tuple[float, float]]:
        w = float(sizes[idx, 0].item())
        h = float(sizes[idx, 1].item())
        x_min, x_max = w*0.5, canvas_w - w*0.5
        y_min, y_max = h*0.5, canvas_h - h*0.5
        bx = min(max(tx, x_min), x_max)
        by = min(max(ty, y_min), y_max)

        if self._is_legal(idx, bx, by, placement, placed_hard,
                          sizes, grid, margin_scale):
            return bx, by

        step_x = max(self.min_search_step, w * 0.35)
        step_y = max(self.min_search_step, h * 0.35)

        for ring in range(1, self.max_rings + 1):
            samples    = 8 + ring * 4
            best       = None
            best_score = float("inf")
            for s in range(samples):
                theta = 2.0 * math.pi * s / samples
                x = min(max(bx + ring*step_x*math.cos(theta), x_min), x_max)
                y = min(max(by + ring*step_y*math.sin(theta), y_min), y_max)
                if not self._is_legal(idx, x, y, placement, placed_hard,
                                      sizes, grid, margin_scale):
                    continue
                score = (x - tx)**2 + (y - ty)**2
                if score < best_score:
                    best_score, best = score, (x, y)
            if best is not None:
                return best
        return None

    def _grid_fallback(
        self, idx, tx, ty, placement, placed_hard,
        sizes, canvas_w, canvas_h, grid,
    ) -> Optional[tuple[float, float]]:
        w = float(sizes[idx, 0].item())
        h = float(sizes[idx, 1].item())
        xs = sorted(
            torch.linspace(w*0.5, canvas_w-w*0.5, self.fallback_grid_steps).tolist(),
            key=lambda v: abs(v - tx),
        )
        ys = sorted(
            torch.linspace(h*0.5, canvas_h-h*0.5, self.fallback_grid_steps).tolist(),
            key=lambda v: abs(v - ty),
        )
        for y in ys:
            for x in xs:
                if self._is_legal(idx, x, y, placement, placed_hard,
                                  sizes, grid, 0.0):
                    return x, y
        return None

    def _legalize(
        self, placement, movable_indices, placed_hard,
        sizes, canvas_w, canvas_h, grid,
    ) -> None:
        for _ in range(2):
            overlapping = [
                idx for idx in movable_indices
                if not self._is_legal(
                    idx,
                    float(placement[idx, 0].item()),
                    float(placement[idx, 1].item()),
                    placement, placed_hard, sizes, grid, 0.0,
                )
            ]
            if not overlapping:
                break
            for idx in overlapping:
                tx = float(placement[idx, 0].item())
                ty = float(placement[idx, 1].item())
                grid.remove(idx, tx, ty)
                pos = self._ring_search(
                    idx, tx, ty, placement, placed_hard,
                    sizes, canvas_w, canvas_h, grid, 0.0,
                )
                if pos is None:
                    pos = self._grid_fallback(
                        idx, tx, ty, placement, placed_hard,
                        sizes, canvas_w, canvas_h, grid,
                    )
                if pos is not None:
                    placement[idx, 0] = pos[0]
                    placement[idx, 1] = pos[1]
                    grid.insert(idx, pos[0], pos[1])
                else:
                    grid.insert(idx, tx, ty)