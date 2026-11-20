"""Time the UniDoc stage using fixed upstream Merizo rows and check TED outputs."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))
from run_fast import method

def keyed(paths):
    rows = {}
    for path in paths:
        for line in path.read_text().splitlines():
            name, _, _ = line.partition('\t')
            if name in rows:
                raise ValueError(f'Duplicate {name} in {path}')
            rows[name] = line
    return rows

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('inputs', type=Path)
    p.add_argument('reference', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--raw-reference', type=Path, nargs='+', required=True)
    p.add_argument('--backends', choices=('fast', 'reference'), nargs='+', required=True)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--batch-size', type=int, default=0)
    args = p.parse_args()
    paths = sorted(args.inputs.resolve().glob('*.pdb'), key=lambda path: (-path.stat().st_size, path.name))
    ref = args.reference.resolve()
    expected = keyed([ref/'chopping_unidoc.txt'])
    expected_raw = keyed(args.raw_reference)
    assert {path.stem for path in paths} == expected.keys() == expected_raw.keys()
    env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               PYTHONDONTWRITEBYTECODE='1')
    rows = []
    root = args.output.resolve()
    root.mkdir()
    step = args.batch_size or len(paths)
    for i, backend in enumerate(args.backends):
        folder = root/f'{i+1:02d}-{backend}'
        folder.mkdir()
        batches = []
        start = time.perf_counter()
        for begin in range(0, len(paths), step):
            batch = folder/f'batch{len(batches):04d}'
            batch.mkdir()
            method('unidoc', paths[begin:begin+step], batch/'unidoc', env, workers=args.workers,
                   inherited=ref/'chopping_merizo.txt', unidoc_backend=backend)
            batches.append(batch)
        wall = time.perf_counter()-start
        filtered = keyed([batch/'chopping_unidoc.txt' for batch in batches])
        raw = keyed([batch/'unidoc/raw.tsv' for batch in batches])
        assert filtered.keys() == raw.keys() == expected.keys()
        row = {'backend': backend, 'wall_seconds': wall, 'inputs': len(paths), 'workers': args.workers,
               'batches': len(batches), 'raw_identical': sum(raw[k] == expected_raw[k] for k in raw),
               'filtered_identical': sum(filtered[k] == expected[k] for k in filtered)}
        rows.append(row)
        (root/'summary.json').write_text(json.dumps(rows, indent=2)+'\n')
        print(row, flush=True)
        for kind, results, reference in [('raw',raw,expected_raw),('filtered',filtered,expected)]:
            if results != reference:
                diff = {k:[reference[k], results[k]] for k in results if results[k] != reference[k]}
                (folder/f'{kind}-differences.json').write_text(json.dumps(diff, indent=2)+'\n')
                raise ValueError(f'{len(diff)} {kind} differences; see {folder}')

if __name__ == '__main__':
    main()
