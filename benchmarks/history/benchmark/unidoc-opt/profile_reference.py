"""Time each child command of the deployed UniDoc wrapper on a private copy."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import time

WRAPPER = Path('/home/zhr/tools/ted-tools/ted_consensus_1.0/programs/unidoc/Run_UniDoc_from_scratch_structure_afdb.py')

def instrument(args):
    durations = defaultdict(list)
    original = subprocess.check_output
    def timed(command, *pos, **kwargs):
        name = Path(command.split()[0]).name
        start = time.perf_counter()
        try:
            return original(command, *pos, **kwargs)
        finally:
            durations[name].append(time.perf_counter() - start)
    subprocess.check_output = timed
    sys.argv = [str(WRAPPER), '-l', str(args.inputs), '--out', str(args.output),
                '--inherit_chopping', str(args.merizo)]
    start = time.perf_counter()
    runpy.run_path(str(WRAPPER), run_name='__main__')
    report = {'wrapper_wall': time.perf_counter() - start,
              'commands': {key: {'count': len(vals), 'seconds': sum(vals)} for key, vals in durations.items()}}
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', type=Path)
    parser.add_argument('merizo', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--child', action='store_true')
    args = parser.parse_args()
    if args.child:
        instrument(args)
        return
    args.output.mkdir()
    pdbs = args.output/'pdb'
    pdbs.mkdir()
    paths = sorted(args.inputs.glob('*.pdb'), key=lambda p: (-p.stat().st_size, p.name))
    start = time.perf_counter()
    for path in paths:
        shutil.copy2(path, pdbs/path.name)
    copied = time.perf_counter()
    def run(index):
        subset = args.output/f'inputs{index}.txt'
        subset.write_text(''.join(str((pdbs/p.name).resolve())+'\n' for p in paths[index::4]))
        out = args.output/f'worker{index}.tsv'
        subprocess.run([sys.executable, __file__, '--child', subset, args.merizo, out], check=True)
        return json.loads(out.with_suffix('.json').read_text())
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(run, range(4)))
    commands = defaultdict(lambda: {'count': 0, 'seconds': 0})
    for row in rows:
        for key, vals in row['commands'].items():
            commands[key]['count'] += vals['count']
            commands[key]['seconds'] += vals['seconds']
    report = {'inputs': len(paths), 'copy_wall_seconds': copied-start, 'wall_seconds': time.perf_counter()-start,
              'summed_worker_seconds': sum(r['wrapper_wall'] for r in rows), 'summed_child_commands': dict(commands)}
    (args.output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    shutil.rmtree(pdbs)

if __name__ == '__main__':
    main()
