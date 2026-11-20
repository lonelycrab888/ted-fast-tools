"""Measure an isolated method via the same entry point and filters as run_fast.py."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))
import run_fast

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('method', choices=('merizo','chainsaw'))
    p.add_argument('input', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--variant', default='v2')
    p.add_argument('--gpu', type=int, default=4)
    p.add_argument('--profile', action='store_true')
    args = p.parse_args()
    output = args.output.resolve()
    paths = sorted(args.input.resolve().glob('*.pdb'))
    if not paths:
        p.error('no PDB inputs')
    paths, = run_fast.split_paths(paths, 1)
    gpu_uuid, = run_fast.gpu_ids([args.gpu])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_uuid, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
    output.mkdir(parents=True)
    start = time.perf_counter()
    method_seconds = run_fast.method(args.method, paths, output/args.method, env,
                                     profile=args.profile, variant=args.variant)
    result = dict(wall_seconds=time.perf_counter()-start, method_seconds=method_seconds,
                  method=args.method, variant=args.variant, gpu=args.gpu, inputs=len(paths),
                  profile_enabled=args.profile)
    (output/'timing.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)

if __name__ == '__main__':
    main()
