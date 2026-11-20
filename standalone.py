"""Run one TED method and the same boundary filter used in the consensus pipeline."""
import argparse
import os
from pathlib import Path

from run_fast import gpu_ids, method


def main(name):
    parser = argparse.ArgumentParser(description=f"Run {name} on a single chain A PDB or a directory of PDBs.")
    parser.add_argument("-i", "--input", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path, required=True, help="New result directory.")
    if name == "unidoc":
        mode = parser.add_mutually_exclusive_group(required=True)
        mode.add_argument("--inherit", type=Path, help="Filtered six-column chopping file, keyed by PDB stem.")
        mode.add_argument("--whole-chain", action="store_true", help="Parse the whole input without inherited boundaries.")
        parser.add_argument("--workers", type=int, default=4)
    else:
        parser.add_argument("--gpu", type=int, help="Physical GPU index from nvidia-smi (default: current CUDA visibility).")
    args = parser.parse_args()
    source = args.input.resolve()
    if source.is_dir():
        paths = sorted(p for p in source.glob("*.pdb") if p.is_file())
    elif source.is_file() and source.suffix == ".pdb":
        paths = [source]
    else:
        paths = []
    if not paths:
        parser.error(f"No input PDBs at {source}")
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    options = {}
    if name == "unidoc":
        if args.workers < 1:
            parser.error("--workers must be positive")
        if args.inherit and not args.inherit.is_file():
            parser.error(f"Missing chopping table: {args.inherit}")
        options.update(inherited=args.inherit.resolve() if args.inherit else None,
                       whole_chain=args.whole_chain, workers=args.workers)
    elif args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu_ids([args.gpu])[0]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    method(name, paths, output / name, env, **options)
    print(output / f"chopping_{name}.txt")
