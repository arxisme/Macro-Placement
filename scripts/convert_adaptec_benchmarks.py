#!/usr/bin/env python3
"""Convert Adaptec (Bookshelf) benchmarks to this project's benchmark format.

Output format per design:
  <output_root>/<design>/netlist.pb.txt
  <output_root>/<design>/initial.plc

By default, also validates each converted benchmark with the project loader and
saves a serialized Benchmark tensor to benchmarks/processed/public/<design>.pt.

Example:
  python scripts/convert_adaptec_benchmarks.py \
      --input-root /path/to/adaptec \
      --designs adaptec1 adaptec2 adaptec3 adaptec4 adaptec5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Import MacroPlacement's Bookshelf translator.
_FT_DIR = (
    Path(__file__).resolve().parents[1]
    / "external"
    / "MacroPlacement"
    / "CodeElements"
    / "FormatTranslators"
    / "src"
)
if str(_FT_DIR) not in sys.path:
    sys.path.insert(0, str(_FT_DIR))

from FormatTranslators import BookShelf2ProBufFormat  # type: ignore

from macro_place._plc import PlacementCost
from macro_place.loader import load_benchmark_from_dir


def _required_bookshelf_files(design_dir: Path, design: str) -> list[Path]:
    return [
        design_dir / f"{design}.nodes",
        design_dir / f"{design}.nets",
        design_dir / f"{design}.pl",
        design_dir / f"{design}.scl",
    ]


def convert_design(
    input_root: Path,
    output_root: Path,
    design: str,
    save_pt: bool,
    pt_output_dir: Path,
) -> bool:
    design_src = input_root / design
    if not design_src.exists():
        print(f"  {design}: SKIPPED (missing directory: {design_src})")
        return False

    required = _required_bookshelf_files(design_src, design)
    missing = [p for p in required if not p.exists()]
    if missing:
        print(f"  {design}: FAILED (missing Bookshelf files)")
        for p in missing:
            print(f"    - {p}")
        return False

    design_out = output_root / design
    design_out.mkdir(parents=True, exist_ok=True)
    netlist_pb = design_out / "netlist.pb.txt"
    initial_plc = design_out / "initial.plc"

    # 1) Bookshelf -> protobuf netlist
    BookShelf2ProBufFormat(str(design_src), design, str(netlist_pb))

    # 2) Create an initial .plc from default module positions in the netlist
    plc = PlacementCost(str(netlist_pb))
    meta_info = (
        f"Auto-generated initial placement for {design} from Bookshelf\n"
        f"Columns : 128  Rows : 128\n"
        f"Width : 10000.0  Height : 10000.0\n"
    )
    plc.save_placement(
        str(initial_plc),
        info=meta_info,
    )

    # 3) Validate with this project's loader
    benchmark, _ = load_benchmark_from_dir(str(design_out))

    if save_pt:
        pt_output_dir.mkdir(parents=True, exist_ok=True)
        pt_file = pt_output_dir / f"{design}.pt"
        benchmark.save(str(pt_file))
        print(
            f"  {design}: OK ({benchmark.num_macros} macros, {benchmark.num_nets} nets) -> {design_out} + {pt_file}"
        )
    else:
        print(
            f"  {design}: OK ({benchmark.num_macros} macros, {benchmark.num_nets} nets) -> {design_out}"
        )

    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert Adaptec Bookshelf benchmarks to macro-placement benchmark format."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Root containing design subdirs (e.g., <root>/adaptec1, <root>/adaptec2, ...).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("benchmarks/adaptec"),
        help="Output root for converted benchmark directories.",
    )
    parser.add_argument(
        "--designs",
        nargs="+",
        default=["adaptec1", "adaptec2", "adaptec3", "adaptec4", "adaptec5"],
        help="Design names to convert (must match <design>/<design>.* Bookshelf naming).",
    )
    parser.add_argument(
        "--no-pt",
        action="store_true",
        help="Do not export serialized Benchmark tensors to benchmarks/processed/public.",
    )
    args = parser.parse_args()

    if not _FT_DIR.exists():
        print(f"Error: FormatTranslators not found at {_FT_DIR}")
        print("Run: git submodule update --init external/MacroPlacement")
        return 1

    print("=" * 80)
    print("Converting Adaptec Benchmarks (Bookshelf -> netlist.pb.txt + initial.plc)")
    print("=" * 80)
    print(f"Input root : {args.input_root}")
    print(f"Output root: {args.output_root}")
    print(f"Designs    : {', '.join(args.designs)}")
    print()

    success = 0
    total = len(args.designs)
    pt_output_dir = Path("benchmarks/processed/public")
    for design in args.designs:
        try:
            ok = convert_design(
                input_root=args.input_root,
                output_root=args.output_root,
                design=design,
                save_pt=not args.no_pt,
                pt_output_dir=pt_output_dir,
            )
            if ok:
                success += 1
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"  {design}: FAILED ({exc})")

    print()
    print("=" * 80)
    print(f"Converted {success}/{total} designs")
    print("=" * 80)

    return 0 if success == total else 1


if __name__ == "__main__":
    sys.exit(main())
