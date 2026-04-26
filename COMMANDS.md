# Project Command Reference

This file lists all runnable command surfaces found in this repository.

## 1. Setup and Installation

Run from repository root:

```bash
git clone https://github.com/partcleda/partcl-macro-place-challenge.git
cd partcl-macro-place-challenge
git submodule update --init external/MacroPlacement
uv sync
```

Alternative ORFS setup (optional, from docs):

```bash
cd ..
git clone --depth=1 https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts
cd macro-place-challenge-2026
```

## 2. Main CLI: evaluate

Installed console entry point from pyproject:

```bash
evaluate <placer.py> [options]
```

Common form used in this repo:

```bash
uv run evaluate <placer.py> [options]
```

Direct module form:

```bash
python -m macro_place.evaluate <placer.py> [options]
```

Options:

- `-b`, `--benchmark <name>`: run one benchmark (default `ibm01`)
- `-a`, `--all`: run all IBM benchmarks
- `--ng45`: run NG45 designs (`ariane133`, `ariane136`, `mempool_tile`, `nvdla`)
- `--vis`: save placement images under `vis/`

Examples:

```bash
uv run evaluate submissions/examples/greedy_row_placer.py
uv run evaluate submissions/examples/greedy_row_placer.py -b ibm01
uv run evaluate submissions/examples/greedy_row_placer.py --all
uv run evaluate submissions/examples/greedy_row_placer.py --ng45
uv run evaluate submissions/examples/greedy_row_placer.py --all --vis
```

## 3. Submission Evaluation Commands

```bash
uv run evaluate submissions/examples/simple_random_placer.py
uv run evaluate submissions/examples/simple_random_placer.py --all
uv run evaluate submissions/examples/greedy_row_placer.py
uv run evaluate submissions/examples/greedy_row_placer.py --all
uv run evaluate submissions/examples/greedy_row_placer.py -b ibm03
uv run evaluate submissions/my_heuristic.py -b ibm01
uv run evaluate submissions/my_heuristic.py --all
uv run evaluate submissions/will_seed/placer.py
uv run evaluate submissions/will_seed/placer.py --all
```

## 4. Python Scripts in scripts/

### 4.1 Convert IBM benchmarks

```bash
python scripts/convert_ibm_benchmarks.py
```

### 4.2 Convert ASAP7 benchmarks

```bash
python scripts/convert_asap7_benchmarks.py
```

### 4.3 Generate macro placement TCL

```bash
python scripts/generate_macro_placement_tcl.py [options]
```

Options:

- `--benchmark <name>` (default: `ariane133`)
- `--output <file>` (default: `output/place_macros.tcl`)
- `--seed <int>` (default: `42`)

Example:

```bash
python scripts/generate_macro_placement_tcl.py --benchmark ariane133 --output output/place_macros.tcl --seed 42
```

### 4.4 Evaluate shelfpack placer

```bash
python scripts/evaluate_shelfpack.py --benchmark <name>
python scripts/evaluate_shelfpack.py --benchmark <name> --run-orfs
python scripts/evaluate_shelfpack.py --all --run-orfs
```

Options:

- `--benchmark <name>`
- `--all`
- `--run-orfs`

### 4.5 Evaluate with OpenROAD-flow-scripts (ORFS)

```bash
python scripts/evaluate_with_orfs.py --benchmark <name> [options]
python scripts/evaluate_with_orfs.py --all [options]
```

Options:

- `--benchmark <name>`
- `--all`
- `--orfs-root <path>` (default: `../OpenROAD-flow-scripts`)
- `--output <path>` (default: `output/orfs_evaluation`)
- `--no-docker`
- `--skip-synthesis`
- `--placement <placement.pt>`

Examples:

```bash
python scripts/evaluate_with_orfs.py --benchmark ariane133_ng45 --no-docker
python scripts/evaluate_with_orfs.py --benchmark ariane133_ng45 --no-docker --placement my_placement.pt
python scripts/evaluate_with_orfs.py --all --no-docker
python scripts/evaluate_with_orfs.py --benchmark ariane133_ng45 --orfs-root /path/to/OpenROAD-flow-scripts --no-docker
python scripts/evaluate_with_orfs.py --benchmark ariane133_ng45 --skip-synthesis
```

## 5. OpenROAD TCL Script Commands

```bash
openroad scripts/test_macro_placement.tcl
openroad scripts/run_openroad_flow.tcl
openroad -gui scripts/view_placement_openroad.tcl
```

## 6. Docker Evaluation Commands

### 6.1 Helper script

```bash
./eval_docker/run_eval.sh <team_name> <placer_path> [extra_mount...]
```

Examples from repo:

```bash
./eval_docker/run_eval.sh convex_opt submissions_eval/convex_opt/submissions/dccp_placer.py
./eval_docker/run_eval.sh mtk submissions_eval/mtk_dreamplace_pp/submissions/placer.py submissions_eval/mtk_dreamplace_pp/submissions/dreamplace
```

### 6.2 Direct Docker build/run

```bash
docker build -t macro-place-eval -f eval_docker/Dockerfile .
docker run --rm --network none --gpus all --memory 64g --cpus 16 -v /path/to/placer.py:/submission/placer.py macro-place-eval /submission/placer.py --all
```

## 7. Package Module Demo Entrypoints

`macro_place/def_writer.py` has a direct module demo block:

```bash
python -m macro_place.def_writer
```

## 8. Testing and Dev Commands

```bash
pytest
pytest -v
pytest test/test_smoke.py
python -m pytest
```

Optional dev tools listed in pyproject optional deps:

```bash
black .
flake8 .
pytest --cov
```

## 9. Notes

- All commands assume you run from repository root unless stated otherwise.
- Some commands require initialized submodules (`external/MacroPlacement`) and optional ORFS checkout.
- Several docs mention helper scripts such as `scripts/demo_placement_to_def.py`, `scripts/compute_ibm_baselines.py`, and `scripts/generate_leaderboard.py`; these files are not present in this workspace snapshot.

## Validate the placement:
```
uv run evaluate submissions/cluster_anchor_placer.py -b ibm01 --vis
pythonfrom macro_place.utils import validate_placement
is_valid, violations = validate_placement(placement, benchmark)
print(is_valid, violations)
bash# 3. For OpenROAD DEF export (optional, takes 3-8 hrs per design)
python scripts/evaluate_with_orfs.py --benchmark ariane133_ng45 --no-docker \
    --placement my_placement.pt
```