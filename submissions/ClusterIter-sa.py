"""Cluster-and-anchor macro placer with pairwise swap improvement.

Algorithm
---------
Phase 1 — Greedy placement (fast, one pass)
    - Union-Find clusters of connected macros
    - Golden-angle spiral anchors per cluster
    - Place macros largest-first using ring search for legality
    - SpatialGrid makes legality checks O(1) instead of O(n)

Phase 2 — Pairwise swap improvement (iterative)
    - For every pair of movable macros, try swapping their positions
    - Keep the swap if it reduces total HPWL
    - Repeat until no swap improves the cost (or max_swap_rounds hit)
    - This is what actually "iterates" and produces visible improvement

Usage:
    uv run evaluate submissions/cluster_anchor_placer.py -b ibm01
    uv run evaluate submissions/cluster_anchor_placer.py --all
"""

from __future__ import annotations

import math
import time
from typing import Optional

import torch

from macro_place.benchmark import Benchmark


# ---------------------------------------------------------------------------
# Spatial grid — O(1) overlap queries
# ---------------------------------------------------------------------------

class SpatialGrid:
    """
    Buckets macros into canvas cells for fast neighbour lookup.
    Overlap queries only check the 3x3 neighbourhood of a position.
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
# HPWL helpers
# ---------------------------------------------------------------------------

def _build_net_index(benchmark: Benchmark) -> list[list[int]]:
    """
    For each macro, return the list of net indices it appears in.
    Used to compute HPWL delta efficiently (only re-evaluate affected nets).
    """
    n = benchmark.num_hard_macros + len(benchmark.net_nodes)  # upper bound
    macro_to_nets: list[list[int]] = [
        [] for _ in range(benchmark.num_hard_macros)
    ]
    for net_idx, net in enumerate(benchmark.net_nodes):
        for node in net.tolist():
            if node < benchmark.num_hard_macros:
                macro_to_nets[node].append(net_idx)
    return macro_to_nets


def _net_hpwl(net: torch.Tensor, positions: torch.Tensor) -> float:
    """HPWL of a single net given current positions."""
    nodes = net[net < len(positions)]
    if nodes.numel() < 2:
        return 0.0
    pts  = positions[nodes]
    return float((pts[:, 0].max() - pts[:, 0].min()
                + pts[:, 1].max() - pts[:, 1].min()).item())


def _total_hpwl(benchmark: Benchmark, positions: torch.Tensor) -> float:
    total = 0.0
    for net in benchmark.net_nodes:
        total += _net_hpwl(net, positions)
    return total


def _delta_hpwl_swap(
    ia: int, ib: int,
    benchmark: Benchmark,
    positions: torch.Tensor,
    macro_to_nets: list[list[int]],
) -> float:
    """
    Compute the HPWL change from swapping positions of macro ia and ib.
    Only re-evaluates nets that contain ia or ib — much faster than
    recomputing all nets.

    Returns delta = new_hpwl - old_hpwl.
    Negative delta means the swap improves (reduces) wirelength.
    """
    # Nets affected by either macro
    affected = set(macro_to_nets[ia]) | set(macro_to_nets[ib])
    if not affected:
        return 0.0

    nets = benchmark.net_nodes

    # HPWL before swap
    before = sum(_net_hpwl(nets[n], positions) for n in affected)

    # Temporarily swap
    pos_a = positions[ia].clone()
    pos_b = positions[ib].clone()
    positions[ia] = pos_b
    positions[ib] = pos_a

    # HPWL after swap
    after = sum(_net_hpwl(nets[n], positions) for n in affected)

    # Restore
    positions[ia] = pos_a
    positions[ib] = pos_b

    return after - before


# ---------------------------------------------------------------------------
# Main placer
# ---------------------------------------------------------------------------

class ClusterAnchorPlacer:
    """
    Phase 1: greedy cluster-and-anchor placement (fast, one pass).
    Phase 2: pairwise swap improvement loop (iterative, visible improvement).
    """

    def __init__(
        self,
        cluster_size: int        = 8,
        max_rings: int           = 60,
        min_search_step: float   = 2.0,
        fallback_grid_steps: int = 28,
        edge_gap: float          = 0.001,
        # Swap improvement parameters
        max_swap_rounds: int     = 20,    # max improvement iterations
        max_swap_time_s: float   = 120.0, # time budget for swap phase (seconds)
    ):
        self.cluster_size        = max(1, int(cluster_size))
        self.max_rings           = max(1, int(max_rings))
        self.min_search_step     = float(min_search_step)
        self.fallback_grid_steps = max(4, int(fallback_grid_steps))
        self.edge_gap            = float(edge_gap)
        self.max_swap_rounds     = max_swap_rounds
        self.max_swap_time_s     = max_swap_time_s

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        t0 = time.time()

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

        # Build spatial grid
        max_dim   = float(sizes[movable_indices].max().item())
        cell_size = max(max_dim, self.min_search_step)
        grid      = SpatialGrid(canvas_w, canvas_h, cell_size)

        # Seed grid with fixed macros
        placed_hard = torch.zeros(num_hard, dtype=torch.bool)
        placed_hard[~movable_mask[:num_hard]] = True
        for idx in torch.where(placed_hard)[0].tolist():
            grid.insert(idx,
                        float(placement[idx, 0].item()),
                        float(placement[idx, 1].item()))

        # Cluster and anchor
        degrees  = self._macro_degrees(benchmark)
        clusters = self._build_clusters(benchmark, movable_mask, degrees)
        clusters.sort(key=lambda cl: (
            -sum(float(degrees[i].item()) for i in cl),
            -sum(float((sizes[i,0]*sizes[i,1]).item()) for i in cl),
            -len(cl),
        ))
        cluster_of = {idx: cid for cid, cl in enumerate(clusters) for idx in cl}
        anchors    = self._cluster_anchors(len(clusters), canvas_w, canvas_h)

        # Largest-first order
        all_sorted = sorted(
            movable_indices,
            key=lambda i: (
                -float((sizes[i,0]*sizes[i,1]).item()),
                -float(sizes[i,1].item()),
                -float(sizes[i,0].item()),
            ),
        )

        # ----------------------------------------------------------------
        # PHASE 1 — Greedy placement
        # ----------------------------------------------------------------
        print(f"[Placer] Phase 1: greedy placement of {len(all_sorted)} macros...")
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

        self._legalize(placement, all_sorted, placed_hard,
                       sizes, canvas_w, canvas_h, grid)

        hpwl_after_greedy = _total_hpwl(benchmark, placement)
        print(f"[Placer] Phase 1 done.  HPWL={hpwl_after_greedy:.2f}  "
              f"time={time.time()-t0:.1f}s")

        # ----------------------------------------------------------------
        # PHASE 2 — Pairwise swap improvement
        # ----------------------------------------------------------------
        print(f"[Placer] Phase 2: pairwise swap improvement "
              f"(max {self.max_swap_rounds} rounds, "
              f"{self.max_swap_time_s}s budget)...")
        # PHASE 2 — Simulated Annealing
        macro_to_nets = _build_net_index(benchmark)
        cur_hpwl = self._simulated_annealing(
            all_sorted, benchmark, placement,
            sizes, canvas_w, canvas_h, grid,
            placed_hard, macro_to_nets,
            t_swap := time.time(),
        )

        # macro_to_nets = _build_net_index(benchmark)
        # t_swap        = time.time()
        # cur_hpwl      = hpwl_after_greedy

        # for round_idx in range(self.max_swap_rounds):
        #     if time.time() - t_swap > self.max_swap_time_s:
        #         print(f"[Placer] Swap time budget reached at round {round_idx}.")
        #         break

        #     swaps_this_round = 0

        #     for i in range(len(all_sorted)):
        #         for j in range(i + 1, len(all_sorted)):
        #             ia = all_sorted[i]
        #             ib = all_sorted[j]

        #             delta = _delta_hpwl_swap(
        #                 ia, ib, benchmark, placement, macro_to_nets
        #             )

        #             if delta >= 0:
        #                 continue  # no improvement, skip

        #             # Check both positions are still legal after swap
        #             xa, ya = float(placement[ia,0].item()), float(placement[ia,1].item())
        #             xb, yb = float(placement[ib,0].item()), float(placement[ib,1].item())

        #             # Temporarily remove both from grid
        #             grid.remove(ia, xa, ya)
        #             grid.remove(ib, xb, yb)

        #             # Check if each macro is legal at the other's position
        #             legal_a = self._is_legal(ia, xb, yb, placement,
        #                                      placed_hard, sizes, grid)
        #             legal_b = self._is_legal(ib, xa, ya, placement,
        #                                      placed_hard, sizes, grid)

        #             if legal_a and legal_b:
        #                 # Accept swap
        #                 placement[ia, 0] = xb
        #                 placement[ia, 1] = yb
        #                 placement[ib, 0] = xa
        #                 placement[ib, 1] = ya
        #                 grid.insert(ia, xb, yb)
        #                 grid.insert(ib, xa, ya)
        #                 cur_hpwl += delta
        #                 swaps_this_round += 1
        #             else:
        #                 # Revert — put back in grid
        #                 grid.insert(ia, xa, ya)
        #                 grid.insert(ib, xb, yb)

        #     print(f"[Placer] Round {round_idx+1:2d}: "
        #           f"swaps={swaps_this_round:4d}  "
        #           f"HPWL={cur_hpwl:.2f}  "
        #           f"improvement={100*(hpwl_after_greedy-cur_hpwl)/hpwl_after_greedy:.2f}%  "
        #           f"time={time.time()-t0:.1f}s")

        #     if swaps_this_round == 0:
        #         print(f"[Placer] Converged after {round_idx+1} rounds.")
        #         break

        # print(f"\n[Placer] Done.  "
        #       f"HPWL: {hpwl_after_greedy:.2f} → {cur_hpwl:.2f}  "
        #       f"({100*(hpwl_after_greedy-cur_hpwl)/hpwl_after_greedy:.2f}% improvement)  "
        #       f"total time={time.time()-t0:.1f}s")

        return placement

    # ------------------------------------------------------------------
    # Clustering helpers
    # ------------------------------------------------------------------

    def _simulated_annealing(
        self,
        all_sorted: list[int],
        benchmark: Benchmark,
        placement: torch.Tensor,
        sizes: torch.Tensor,
        canvas_w: float,
        canvas_h: float,
        grid: SpatialGrid,
        placed_hard: torch.Tensor,
        macro_to_nets: list[list[int]],
        t_start: float,
    ) -> float:
        import random

        n          = len(all_sorted)
        T          = float(sizes[all_sorted].max().item()) * 2.0  # initial temp
        T_min      = T * 1e-4
        alpha      = 0.95                          # cooling rate
        moves_per_step = 10 * n                    # standard rule of thumb
        cur_hpwl   = _total_hpwl(benchmark, placement)

        print(f"[SA] Starting T={T:.2f}, T_min={T_min:.4f}, "
              f"moves/step={moves_per_step}")

        step = 0
        while T > T_min:
            if time.time() - t_start > self.max_swap_time_s:
                print(f"[SA] Time budget hit at step {step}")
                break

            accepted = 0
            for _ in range(moves_per_step):
                # Pick random pair
                ia = random.choice(all_sorted)
                ib = random.choice(all_sorted)
                if ia == ib:
                    continue

                delta = _delta_hpwl_swap(
                    ia, ib, benchmark, placement, macro_to_nets
                )

                # Metropolis criterion
                if delta < 0 or random.random() < math.exp(-delta / T):
                    xa, ya = float(placement[ia,0]), float(placement[ia,1])
                    xb, yb = float(placement[ib,0]), float(placement[ib,1])

                    grid.remove(ia, xa, ya)
                    grid.remove(ib, xb, yb)

                    legal_a = self._is_legal(ia, xb, yb, placement,
                                             placed_hard, sizes, grid)
                    legal_b = self._is_legal(ib, xa, ya, placement,
                                             placed_hard, sizes, grid)

                    if legal_a and legal_b:
                        placement[ia, 0], placement[ia, 1] = xb, yb
                        placement[ib, 0], placement[ib, 1] = xa, ya
                        grid.insert(ia, xb, yb)
                        grid.insert(ib, xa, ya)
                        cur_hpwl += delta
                        accepted += 1
                    else:
                        grid.insert(ia, xa, ya)
                        grid.insert(ib, xb, yb)

            print(f"[SA] step={step:3d}  T={T:.3f}  "
                  f"accepted={accepted}/{moves_per_step}  "
                  f"HPWL={cur_hpwl:.2f}")

            T *= alpha
            step += 1

        return cur_hpwl

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
        cx, cy = canvas_w * 0.5, canvas_h * 0.5
        rx, ry = canvas_w * 0.32, canvas_h * 0.32
        golden = 2.399963229728653
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
            torch.linspace(w*0.5, canvas_w-w*0.5,
                           self.fallback_grid_steps).tolist(),
            key=lambda v: abs(v - tx),
        )
        ys = sorted(
            torch.linspace(h*0.5, canvas_h-h*0.5,
                           self.fallback_grid_steps).tolist(),
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