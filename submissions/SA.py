"""Submission entrypoint for simulated annealing with quick/full profiles.

Use environment variables to switch behavior while keeping evaluator compatibility.

Examples:
- Quick iteration run:
  MP_SA_MODE=quick uv run evaluate submissions/SA.py -b ibm02
- Full run:
  MP_SA_MODE=full uv run evaluate submissions/SA.py -b ibm02

Optional overrides:
- MP_SA_ITERS
- MP_SA_ACTIONS
- MP_SA_TEMP
- MP_SA_SEED
"""

from __future__ import annotations

import os

import torch

from macro_place.benchmark import Benchmark
from submissions.approaches.simulated_annealing import SimulatedAnnealingPlacer


class SAPlacer(SimulatedAnnealingPlacer):
    """Evaluator-discoverable placer class with runtime profiles."""

    def __init__(self):
        mode = os.environ.get("MP_SA_MODE", "quick").strip().lower()

        # Fast profile for day-to-day debugging and quick feedback.
        profiles = {
            "quick": {"num_iters": 2, "num_actions": 1, "max_temperature": 0.5, "seed": 42},
            # Full profile for final evaluation.
            "full": {"num_iters": 10, "num_actions": 2, "max_temperature": 1.0, "seed": 42},
        }
        cfg = profiles.get(mode, profiles["quick"]).copy()

        # Optional per-run tuning without file edits.
        if os.environ.get("MP_SA_ITERS"):
            cfg["num_iters"] = max(1, int(os.environ["MP_SA_ITERS"]))
        if os.environ.get("MP_SA_ACTIONS"):
            cfg["num_actions"] = max(1, int(os.environ["MP_SA_ACTIONS"]))
        if os.environ.get("MP_SA_TEMP"):
            cfg["max_temperature"] = max(1e-6, float(os.environ["MP_SA_TEMP"]))
        if os.environ.get("MP_SA_SEED"):
            cfg["seed"] = int(os.environ["MP_SA_SEED"])

        self._mode = mode
        super().__init__(
            num_iters=cfg["num_iters"],
            num_actions=cfg["num_actions"],
            max_temperature=cfg["max_temperature"],
            seed=cfg["seed"],
        )

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        print(
            "[SA] mode="
            f"{self._mode} "
            f"iters={self.num_iters} actions={self.num_actions} "
            f"temp={self.max_temperature} seed={self.seed}"
        )
        return super().place(benchmark)
