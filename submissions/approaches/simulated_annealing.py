"""
Simulated Annealing Placer (Python adaptation of TILOS SA flow).

This mirrors the high-level structure in:
external/MacroPlacement/CodeElements/SimulatedAnnealing/src/plc_netlist_sa.cpp

Key matching points:
1. Initial macro tiling on grid centers (optionally spiral order)
2. Action set: Swap, Shift, Flip, Move, Shuffle
3. Temperature schedule:
     t = t_max * exp(log(t_min/t_max) * iter / num_iters)
4. Metropolis acceptance:
     accept with probability exp((cur_cost - new_cost) / t)

Notes:
- Hard-macro orientation updates are not exposed in the public Python benchmark API,
  so Flip is implemented as a no-op action.
- Cost is a lightweight Python proxy matching the same weighted form:
    HPWL/norm + 0.5*density + 0.5*congestion
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from macro_place.benchmark import Benchmark


class SimulatedAnnealingPlacer:
    """Python SA placer with move operators aligned to C++ SA baseline."""

    def __init__(
        self,
        action_probs: Sequence[float] = (0.2, 0.2, 0.1, 0.2, 0.3),
        num_actions: int = 1,
        max_temperature: float = 1.0,
        num_iters: int = 8,
        seed: int = 42,
        spiral_flag: bool = True,
        smooth_factor: int = 2,
    ):
        if len(action_probs) != 5:
            raise ValueError("action_probs must have 5 entries: swap, shift, flip, move, shuffle")
        if sum(action_probs) <= 0.0:
            raise ValueError("action_probs must sum to > 0")

        self.action_probs = np.array(action_probs, dtype=np.float64)
        self.action_probs = self.action_probs / self.action_probs.sum()
        self.cum_probs = np.cumsum(self.action_probs)

        self.num_actions = max(1, int(num_actions))
        self.max_temperature = float(max_temperature)
        self.num_iters = max(1, int(num_iters))
        self.seed = int(seed)
        self.spiral_flag = bool(spiral_flag)
        self.smooth_factor = max(0, int(smooth_factor))

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        placement = benchmark.macro_positions.clone()

        hard_mask = benchmark.get_hard_macro_mask()
        movable = benchmark.get_movable_mask() & hard_mask
        macros = torch.where(movable)[0].tolist()
        if len(macros) == 0:
            return placement

        rng = random.Random(self.seed)

        grid_w = float(benchmark.canvas_width) / max(1, int(benchmark.grid_cols))
        grid_h = float(benchmark.canvas_height) / max(1, int(benchmark.grid_rows))

        self._init_macro_placement(placement, benchmark, macros, grid_w, grid_h, self.spiral_flag)

        norm_hpwl = self._norm_hpwl(benchmark)
        cur_cost = self._calc_cost(placement, benchmark, norm_hpwl, grid_w, grid_h)
        best_cost = cur_cost
        best_placement = placement.clone()

        n_steps = self.num_actions * len(macros)
        t_min = 1e-8
        t_max = max(self.max_temperature, 1e-6)
        t_factor = math.log(t_min / t_max)

        for it in range(self.num_iters):
            temperature = t_max * math.exp(t_factor * float(it) / float(self.num_iters))

            for _ in range(n_steps):
                changed, backup = self._action(placement, benchmark, macros, grid_w, grid_h, rng)
                if not changed:
                    continue

                new_cost = self._calc_cost(placement, benchmark, norm_hpwl, grid_w, grid_h)
                if new_cost < cur_cost:
                    cur_cost = new_cost

                accept_prob = math.exp((cur_cost - new_cost) / max(temperature, 1e-12))
                if accept_prob < rng.random():
                    self._restore(placement, backup)
                else:
                    cur_cost = new_cost
                    if cur_cost < best_cost:
                        best_cost = cur_cost
                        best_placement = placement.clone()

        return best_placement

    def _init_macro_placement(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        macros: List[int],
        grid_w: float,
        grid_h: float,
        spiral_flag: bool,
    ):
        n_cols = max(1, int(benchmark.grid_cols))
        n_rows = max(1, int(benchmark.grid_rows))
        grid_order = self._grid_order(n_cols, n_rows, spiral_flag)

        # Sort by area descending, matching C++ InitMacroPlacement.
        sorted_macros = sorted(
            macros,
            key=lambda idx: -float((benchmark.macro_sizes[idx, 0] * benchmark.macro_sizes[idx, 1]).item()),
        )

        occupied = [False] * (n_rows * n_cols)
        placed_set = set()

        # Fixed hard macros are immutable overlap obstacles.
        fixed_hard = torch.where(benchmark.macro_fixed[: benchmark.num_hard_macros])[0].tolist()
        for m in fixed_hard:
            placed_set.add(int(m))

        for macro in sorted_macros:
            for grid_id in grid_order:
                if occupied[grid_id]:
                    continue
                row_id = grid_id // n_cols
                col_id = grid_id % n_cols
                x = (col_id + 0.5) * grid_w
                y = (row_id + 0.5) * grid_h

                old_x = float(placement[macro, 0].item())
                old_y = float(placement[macro, 1].item())
                placement[macro, 0] = x
                placement[macro, 1] = y

                if self._is_feasible_against(placement, benchmark, macro, placed_set):
                    occupied[grid_id] = True
                    placed_set.add(macro)
                    break

                placement[macro, 0] = old_x
                placement[macro, 1] = old_y

    def _grid_order(self, n_cols: int, n_rows: int, spiral_flag: bool) -> List[int]:
        if not spiral_flag:
            return list(range(n_cols * n_rows))

        visited = [False] * (n_cols * n_rows)
        order: List[int] = []
        dir_row = [0, 1, 0, -1]
        dir_col = [1, 0, -1, 0]
        row = 0
        col = 0
        d = 0

        for _ in range(n_cols * n_rows):
            grid_id = row * n_cols + col
            visited[grid_id] = True
            order.append(grid_id)

            nr = row + dir_row[d]
            nc = col + dir_col[d]
            if 0 <= nr < n_rows and 0 <= nc < n_cols and not visited[nr * n_cols + nc]:
                row = nr
                col = nc
            else:
                d = (d + 1) % 4
                row += dir_row[d]
                col += dir_col[d]

        return order

    def _action(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        macros: List[int],
        grid_w: float,
        grid_h: float,
        rng: random.Random,
    ) -> Tuple[bool, Dict[int, Tuple[float, float]]]:
        p = rng.random()
        if p <= self.cum_probs[0]:
            return self._swap(placement, benchmark, macros, rng)
        if p <= self.cum_probs[1]:
            return self._shift(placement, benchmark, macros, grid_w, grid_h, rng)
        if p <= self.cum_probs[2]:
            return self._flip(placement, macros, rng)
        if p <= self.cum_probs[3]:
            return self._move(placement, benchmark, macros, grid_w, grid_h, rng)
        return self._shuffle(placement, benchmark, macros, rng)

    def _swap(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        macros: List[int],
        rng: random.Random,
    ) -> Tuple[bool, Dict[int, Tuple[float, float]]]:
        if len(macros) < 2:
            return False, {}

        for _ in range(5):
            a = rng.choice(macros)
            b = rng.choice(macros)
            if a == b:
                continue

            backup = {
                a: (float(placement[a, 0].item()), float(placement[a, 1].item())),
                b: (float(placement[b, 0].item()), float(placement[b, 1].item())),
            }
            placement[a, 0], placement[a, 1] = backup[b][0], backup[b][1]
            placement[b, 0], placement[b, 1] = backup[a][0], backup[a][1]

            if self._is_feasible(placement, benchmark, a) and self._is_feasible(placement, benchmark, b):
                return True, backup

            self._restore(placement, backup)

        return False, {}

    def _shift(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        macros: List[int],
        grid_w: float,
        grid_h: float,
        rng: random.Random,
    ) -> Tuple[bool, Dict[int, Tuple[float, float]]]:
        macro = rng.choice(macros)
        x0 = float(placement[macro, 0].item())
        y0 = float(placement[macro, 1].item())
        backup = {macro: (x0, y0)}

        neighbors = [(x0 - grid_w, y0), (x0 + grid_w, y0), (x0, y0 - grid_h), (x0, y0 + grid_h)]
        rng.shuffle(neighbors)

        for x, y in neighbors:
            placement[macro, 0] = x
            placement[macro, 1] = y
            if self._is_feasible(placement, benchmark, macro):
                return True, backup

        self._restore(placement, backup)
        return False, {}

    def _move(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        macros: List[int],
        grid_w: float,
        grid_h: float,
        rng: random.Random,
    ) -> Tuple[bool, Dict[int, Tuple[float, float]]]:
        macro = rng.choice(macros)
        backup = {macro: (float(placement[macro, 0].item()), float(placement[macro, 1].item()))}

        col = rng.randrange(max(1, int(benchmark.grid_cols)))
        row = rng.randrange(max(1, int(benchmark.grid_rows)))
        placement[macro, 0] = (col + 0.5) * grid_w
        placement[macro, 1] = (row + 0.5) * grid_h

        if self._is_feasible(placement, benchmark, macro):
            return True, backup

        self._restore(placement, backup)
        return False, {}

    def _shuffle(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        macros: List[int],
        rng: random.Random,
    ) -> Tuple[bool, Dict[int, Tuple[float, float]]]:
        if len(macros) < 4:
            return False, {}

        picked = rng.sample(macros, 4)
        backup = {m: (float(placement[m, 0].item()), float(placement[m, 1].item())) for m in picked}
        locs = list(backup.values())
        rng.shuffle(locs)

        for i, macro in enumerate(picked):
            placement[macro, 0] = locs[i][0]
            placement[macro, 1] = locs[i][1]

        for macro in picked:
            if not self._is_feasible(placement, benchmark, macro):
                self._restore(placement, backup)
                return False, {}

        return True, backup

    def _flip(
        self,
        _placement: torch.Tensor,
        _macros: List[int],
        _rng: random.Random,
    ) -> Tuple[bool, Dict[int, Tuple[float, float]]]:
        # Orientation is not exposed in this benchmark API; keep action slot for parity.
        return False, {}

    def _restore(self, placement: torch.Tensor, backup: Dict[int, Tuple[float, float]]):
        for macro, (x, y) in backup.items():
            placement[macro, 0] = x
            placement[macro, 1] = y

    def _is_feasible(self, placement: torch.Tensor, benchmark: Benchmark, macro: int) -> bool:
        if not self._inside_canvas(placement, benchmark, macro):
            return False

        num_hard = benchmark.num_hard_macros
        for other in range(num_hard):
            if other == macro:
                continue
            if self._overlap(placement, benchmark, macro, other):
                return False
        return True

    def _is_feasible_against(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        macro: int,
        obstacles: set[int],
    ) -> bool:
        if not self._inside_canvas(placement, benchmark, macro):
            return False

        for other in obstacles:
            if other == macro:
                continue
            if self._overlap(placement, benchmark, macro, other):
                return False
        return True

    def _inside_canvas(self, placement: torch.Tensor, benchmark: Benchmark, macro: int) -> bool:
        x = float(placement[macro, 0].item())
        y = float(placement[macro, 1].item())
        w = float(benchmark.macro_sizes[macro, 0].item())
        h = float(benchmark.macro_sizes[macro, 1].item())

        return (
            x - 0.5 * w >= 0.0
            and y - 0.5 * h >= 0.0
            and x + 0.5 * w <= float(benchmark.canvas_width)
            and y + 0.5 * h <= float(benchmark.canvas_height)
        )

    def _overlap(self, placement: torch.Tensor, benchmark: Benchmark, a: int, b: int) -> bool:
        xa = float(placement[a, 0].item())
        ya = float(placement[a, 1].item())
        wa = float(benchmark.macro_sizes[a, 0].item())
        ha = float(benchmark.macro_sizes[a, 1].item())

        xb = float(placement[b, 0].item())
        yb = float(placement[b, 1].item())
        wb = float(benchmark.macro_sizes[b, 0].item())
        hb = float(benchmark.macro_sizes[b, 1].item())

        sep_x = 0.5 * (wa + wb)
        sep_y = 0.5 * (ha + hb)
        return abs(xa - xb) <= sep_x and abs(ya - yb) <= sep_y

    def _norm_hpwl(self, benchmark: Benchmark) -> float:
        if benchmark.num_nets == 0:
            return 1.0
        net_w = benchmark.net_weights
        total_w = float(torch.sum(net_w).item()) if net_w.numel() > 0 else float(benchmark.num_nets)
        norm = total_w * (float(benchmark.canvas_width) + float(benchmark.canvas_height))
        return max(norm, 1e-6)

    def _calc_cost(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        norm_hpwl: float,
        grid_w: float,
        grid_h: float,
    ) -> float:
        hpwl = self._hpwl(placement, benchmark)
        density = self._density_cost(placement, benchmark, grid_w, grid_h)
        congestion = self._congestion_cost(placement, benchmark, grid_w, grid_h)
        return hpwl / norm_hpwl + 0.5 * density + 0.5 * congestion

    def _hpwl(self, placement: torch.Tensor, benchmark: Benchmark) -> float:
        if benchmark.num_nets == 0:
            return 0.0

        total = 0.0
        for net_nodes in benchmark.net_nodes:
            if net_nodes.numel() <= 1:
                continue
            xs: List[float] = []
            ys: List[float] = []
            for owner in net_nodes.tolist():
                x, y = self._owner_xy(placement, benchmark, int(owner))
                xs.append(x)
                ys.append(y)
            total += (max(xs) - min(xs)) + (max(ys) - min(ys))
        return total

    def _owner_xy(self, placement: torch.Tensor, benchmark: Benchmark, owner_idx: int) -> Tuple[float, float]:
        if owner_idx < benchmark.num_macros:
            return float(placement[owner_idx, 0].item()), float(placement[owner_idx, 1].item())
        port_id = owner_idx - benchmark.num_macros
        if 0 <= port_id < benchmark.port_positions.shape[0]:
            return (
                float(benchmark.port_positions[port_id, 0].item()),
                float(benchmark.port_positions[port_id, 1].item()),
            )
        return 0.0, 0.0

    def _density_cost(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        grid_w: float,
        grid_h: float,
    ) -> float:
        rows = max(1, int(benchmark.grid_rows))
        cols = max(1, int(benchmark.grid_cols))
        density = np.zeros((rows, cols), dtype=np.float64)
        cell_area = max(grid_w * grid_h, 1e-12)

        for i in range(benchmark.num_hard_macros):
            x = float(placement[i, 0].item())
            y = float(placement[i, 1].item())
            w = float(benchmark.macro_sizes[i, 0].item())
            h = float(benchmark.macro_sizes[i, 1].item())
            lx = x - 0.5 * w
            ly = y - 0.5 * h
            ux = x + 0.5 * w
            uy = y + 0.5 * h

            lx_id = max(0, min(cols - 1, int(math.floor(lx / grid_w))))
            ux_id = max(0, min(cols - 1, int(math.floor((ux - 1e-9) / grid_w))))
            ly_id = max(0, min(rows - 1, int(math.floor(ly / grid_h))))
            uy_id = max(0, min(rows - 1, int(math.floor((uy - 1e-9) / grid_h))))

            for ry in range(ly_id, uy_id + 1):
                cell_ly = ry * grid_h
                cell_uy = (ry + 1) * grid_h
                ov_y = max(0.0, min(uy, cell_uy) - max(ly, cell_ly))
                if ov_y <= 0.0:
                    continue
                for cx in range(lx_id, ux_id + 1):
                    cell_lx = cx * grid_w
                    cell_ux = (cx + 1) * grid_w
                    ov_x = max(0.0, min(ux, cell_ux) - max(lx, cell_lx))
                    if ov_x <= 0.0:
                        continue
                    density[ry, cx] += (ov_x * ov_y) / cell_area

        flat = np.sort(density.reshape(-1))[::-1]
        top_k = max(1, int(math.floor(rows * cols * 0.1)))
        # C++ scales this term internally by 0.5, then SA uses another 0.5 multiplier.
        return float(np.mean(flat[:top_k]) * 0.5)

    def _congestion_cost(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        grid_w: float,
        grid_h: float,
    ) -> float:
        rows = max(1, int(benchmark.grid_rows))
        cols = max(1, int(benchmark.grid_cols))
        hor = np.zeros((rows, cols), dtype=np.float64)
        ver = np.zeros((rows, cols), dtype=np.float64)
        mhor = np.zeros((rows, cols), dtype=np.float64)
        mver = np.zeros((rows, cols), dtype=np.float64)

        net_weights = benchmark.net_weights
        for net_id, net_nodes in enumerate(benchmark.net_nodes):
            if net_nodes.numel() <= 1:
                continue

            pts = [self._owner_xy(placement, benchmark, int(owner)) for owner in net_nodes.tolist()]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            min_c = max(0, min(cols - 1, int(math.floor(min(xs) / grid_w))))
            max_c = max(0, min(cols - 1, int(math.floor(max(xs) / grid_w))))
            min_r = max(0, min(rows - 1, int(math.floor(min(ys) / grid_h))))
            max_r = max(0, min(rows - 1, int(math.floor(max(ys) / grid_h))))

            span_c = max(1, max_c - min_c + 1)
            span_r = max(1, max_r - min_r + 1)
            area = span_c * span_r
            weight = float(net_weights[net_id].item()) if net_id < net_weights.shape[0] else 1.0

            h_amt = weight * float(span_c - 1) / float(area)
            v_amt = weight * float(span_r - 1) / float(area)
            hor[min_r : max_r + 1, min_c : max_c + 1] += h_amt
            ver[min_r : max_r + 1, min_c : max_c + 1] += v_amt

        # Approximate macro-induced routing blockages from overlap lengths.
        for i in range(benchmark.num_hard_macros):
            x = float(placement[i, 0].item())
            y = float(placement[i, 1].item())
            w = float(benchmark.macro_sizes[i, 0].item())
            h = float(benchmark.macro_sizes[i, 1].item())
            lx = x - 0.5 * w
            ly = y - 0.5 * h
            ux = x + 0.5 * w
            uy = y + 0.5 * h

            lx_id = max(0, min(cols - 1, int(math.floor(lx / grid_w))))
            ux_id = max(0, min(cols - 1, int(math.floor((ux - 1e-9) / grid_w))))
            ly_id = max(0, min(rows - 1, int(math.floor(ly / grid_h))))
            uy_id = max(0, min(rows - 1, int(math.floor((uy - 1e-9) / grid_h))))

            for ry in range(ly_id, uy_id + 1):
                cell_ly = ry * grid_h
                cell_uy = (ry + 1) * grid_h
                ov_y = max(0.0, min(uy, cell_uy) - max(ly, cell_ly))
                if ov_y <= 0.0:
                    continue
                for cx in range(lx_id, ux_id + 1):
                    cell_lx = cx * grid_w
                    cell_ux = (cx + 1) * grid_w
                    ov_x = max(0.0, min(ux, cell_ux) - max(lx, cell_lx))
                    if ov_x <= 0.0:
                        continue
                    mhor[ry, cx] += ov_y
                    mver[ry, cx] += ov_x

        # Smooth congestion like C++ implementation.
        smooth_h = np.zeros_like(hor)
        smooth_v = np.zeros_like(ver)
        sf = self.smooth_factor
        for r in range(rows):
            for c in range(cols):
                v_start = max(0, c - sf)
                v_end = min(cols - 1, c + sf)
                v_share = ver[r, c] / float(v_end - v_start + 1)
                smooth_v[r, v_start : v_end + 1] += v_share

                h_start = max(0, r - sf)
                h_end = min(rows - 1, r + sf)
                h_share = hor[r, c] / float(h_end - h_start + 1)
                smooth_h[h_start : h_end + 1, c] += h_share

        cong_vals: List[float] = []
        v_cap = max(grid_w * float(benchmark.vroutes_per_micron), 1e-12)
        h_cap = max(grid_h * float(benchmark.hroutes_per_micron), 1e-12)
        for r in range(rows):
            for c in range(cols):
                cong_vals.append((smooth_v[r, c] + mver[r, c]) / v_cap)
                cong_vals.append((smooth_h[r, c] + mhor[r, c]) / h_cap)

        cong_vals.sort(reverse=True)
        top_k = max(1, int(math.floor(rows * cols * 0.1)))
        return float(sum(cong_vals[:top_k]) / float(top_k))
