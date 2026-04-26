
# ClusterAnchor

This project focuses on the intricate challenge of macro placement. The main algorithm currently under active development is implemented in `submissions/ClusterIter.py`. The `Results` folder contains the final output benchmarks, along with physical floorplan visualizations stored in `Results/Cluster-Vis`.

## Running the Placer

You can evaluate the main algorithm using the project's native evaluator. To test on the ICCAD04 `ibm01` benchmark:
```bash
uv run evaluate submissions/ClusterIter.py -b ibm01
```

You can seamlessly evaluate on the gigantic Adaptec benchmarks by specifying the `-b` flag:
```bash
uv run evaluate submissions/ClusterIter.py -b adaptec1
```

## Parsing Adaptec Benchmarks to `.pt` Tensors
No need to convert now , the files are already converted.
If you are working with the original raw `.tar.gz` Bookshelf formats from the ISPD2005 dataset, they are converted into `.pt` serialized tensors and `.pb.txt` protobuf files via the `scripts/convert_adaptec_benchmarks.py` script. 

First, ensure the uncompressed adaptec directories (e.g., `adaptec1/`) are located in `benchmarks/ispd2005`. Then, execute the following command:
```bash
uv run python scripts/convert_adaptec_benchmarks.py --input-root benchmarks/ispd2005 --designs adaptec1 adaptec2 adaptec3 adaptec4
```
This script handles the Bookshelf parsing, resolves internal macro pin translation rules, patches grid geometry metadata into the generated `.plc`, and successfully outputs everything to `benchmarks/adaptec/` and `benchmarks/processed/public/` (the `.pt` models).

## About Macro Placement

Macro placement is the problem of positioning large fixed-size blocks (SRAMs, IPs, analog macros, etc.) on a chip floorplan so that routing congestion, timing, power delivery, and area constraints are balanced. Unlike standard-cell placement, macros have strong geometric and connectivity constraints, so the challenge is to explore a highly discrete design space while minimizing wirelength, avoiding blockages, and preserving downstream routability and timing quality.

For example, the **ibm01** benchmark has:
- **246 hard macros** of varying sizes (ranging from 0.8 to 27 μm², with 33× size variation)
- **7,269 nets** connecting macros to each other and to 894 pre-placed standard cell clusters
- **A 22.9 × 23.0 μm canvas** with 42.8% area utilization




Evaluation is two-tiered:

## 🚀 Quick Start

### Installation 

```bash
# Clone the repository
git clone https://github.com/partcleda/partcl-macro-place-challenge.git
cd partcl-macro-place-challenge

# Initialize TILOS MacroPlacement submodule (required for evaluation)
git submodule update --init external/MacroPlacement

# Install the package and all dependencies
uv sync

# Verify the setup
uv run evaluate submissions/examples/greedy_row_placer.py -b ibm01
```

### Run Your First Example

```bash
# Run the greedy row placer on ibm01
uv run evaluate submissions/examples/greedy_row_placer.py

# Run on all 17 IBM benchmarks
uv run evaluate submissions/examples/greedy_row_placer.py --all

# Run on NG45 commercial designs (ariane133, ariane136, mempool_tile, nvdla)
uv run evaluate submissions/examples/greedy_row_placer.py --ng45

# Visualize the result
uv run evaluate submissions/examples/greedy_row_placer.py --vis
uv run evaluate submissions/examples/greedy_row_placer.py --all --vis
```

Running on all benchmarks produces a summary like:
```
Benchmark     Proxy        SA   RePlAce     vs SA  vs RePlAce  Overlaps
   ibm01    2.0463    1.3166    0.9976    -55.4%     -105.1%         0
   ibm02    2.0431    1.9072    1.8370     -7.1%      -11.2%         0
   ...
     AVG    2.2109    2.1251    1.4578     -4.0%      -51.7%         0
```

The greedy placer achieves zero overlaps but makes no attempt to optimize wirelength or connectivity — your job is to do better! See [`SETUP.md`](SETUP.md) for the full API reference and [`submissions/examples/`](submissions/examples/) for working examples.

## 🎯 IBM Benchmark Suite (ICCAD04)

We evaluate on the complete ICCAD04 IBM benchmark suite:

| Benchmark | Macros | Nets | Canvas (μm) | Area Util. | SA Baseline | RePlAce Baseline |
|-----------|--------|------|-------------|------------|-------------|------------------|
| **ibm01** | 246 | 7,269 | 22.9×23.0 | 42.8% | 1.3166 | **0.9976** ⭐ |
| **ibm02** | 254 | 7,538 | 23.2×23.5 | 43.1% | 1.9072 | **1.8370** ⭐ |
| **ibm03** | 269 | 8,045 | 24.1×24.3 | 44.2% | 1.7401 | **1.3222** ⭐ |
| **ibm04** | 285 | 8,654 | 24.8×25.1 | 44.8% | 1.5037 | **1.3024** ⭐ |
| **ibm06** | 318 | 9,745 | 26.1×26.5 | 46.1% | 2.5057 | **1.6187** ⭐ |
| **ibm07** | 335 | 10,328 | 26.8×27.2 | 46.8% | 2.0229 | **1.4633** ⭐ |
| **ibm08** | 352 | 10,901 | 27.5×27.9 | 47.4% | 1.9239 | **1.4285** ⭐ |
| **ibm09** | 369 | 11,463 | 28.1×28.5 | 48.0% | 1.3875 | **1.1194** ⭐ |
| **ibm10** | 387 | 12,018 | 28.8×29.2 | 48.6% | 2.1108 | **1.5009** ⭐ |
| **ibm11** | 405 | 12,568 | 29.4×29.8 | 49.2% | 1.7111 | **1.1774** ⭐ |
| **ibm12** | 423 | 13,111 | 30.1×30.5 | 49.8% | 2.8261 | **1.7261** ⭐ |
| **ibm13** | 441 | 13,647 | 30.7×31.1 | 50.4% | 1.9141 | **1.3355** ⭐ |
| **ibm14** | 460 | 14,178 | 31.4×31.8 | 51.0% | 2.2750 | **1.5436** ⭐ |
| **ibm15** | 479 | 14,704 | 32.0×32.4 | 51.6% | 2.3000 | **1.5159** ⭐ |
| **ibm16** | 498 | 15,225 | 32.7×33.1 | 52.2% | 2.2337 | **1.4780** ⭐ |
| **ibm17** | 517 | 15,741 | 33.3×33.7 | 52.8% | 3.6726 | **1.6446** ⭐ |
| **ibm18** | 537 | 16,253 | 34.0×34.4 | 53.4% | 2.7755 | **1.7722** ⭐ |

Each benchmark includes:
- Hard macros (you place these)
- Soft macros (you can also place these)
- Nets connecting all components
- Initial placement (hand-crafted, serves as reference)

**Baseline Analysis:**
- RePlAce (⭐) consistently outperforms SA across all benchmarks
- RePlAce achieves 15-55% lower proxy cost than SA
- Both baselines achieve zero overlaps (enforced as hard constraint)



## 📖 Documentation

- **Setup & API Reference**: [`SETUP.md`](SETUP.md) - Infrastructure details, benchmark format, cost computation, testing
- **Example Submissions**: [`submissions/examples/`](submissions/examples/) - Working placer examples

