"""Shard the unchanged UniDoc subprocess over independent PDBs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys

WRAPPER = Path(__file__).resolve().parent / 'upstream/Run_UniDoc_from_scratch_structure_afdb.py'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('inputs', type=Path)
    p.add_argument('merizo', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--workers', type=int, required=True)
    args = p.parse_args()
    paths = args.inputs.read_text().splitlines()
    workers = min(args.workers, len(paths))

    def run_shard(index):
        subset = args.output.with_name(f'inputs-{index}.txt')
        subset.write_text('\n'.join(paths[index::workers])+'\n')
        output = args.output.with_name(f'raw-{index}.tsv')
        code = subprocess.run([sys.executable, WRAPPER, '-l', subset, '--out', output,
                               '--inherit_chopping', args.merizo]).returncode
        if code:
            raise RuntimeError(f'UniDoc shard {index} failed with exit code {code}')
        return output.read_text()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        args.output.write_text(''.join(pool.map(run_shard, range(workers))))


if __name__ == '__main__':
    main()
