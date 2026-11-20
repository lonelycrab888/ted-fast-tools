"""Record concurrent worker call durations for the optimized UniDoc stage."""
from collections import defaultdict
import json
import os
import shlex
from statistics import median
import subprocess
from tempfile import TemporaryDirectory
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parents[1]
sys.path.insert(0, str(BASE))
import unidoc_fast

def startup_probe():
    """Estimate fixed shell, interpreter and CLI overhead on a tiny PDB."""
    with TemporaryDirectory(prefix='ted-unidoc-startup-') as tmp:
        pdb = Path(tmp)/'small.pdb'
        pdb.write_text('END\\n')
        commands = {'python_pass': [sys.executable, '-c', 'pass'],
                    'pdb_tofasta': ['pdb_tofasta', str(pdb)],
                    'pdb_selres': ['pdb_selres', '-1:1', str(pdb)]}
        samples = defaultdict(list)
        for _ in range(20):
            for name, command in commands.items():
                start = time.perf_counter()
                subprocess.run(shlex.join(command), shell=True, check=True, stdout=subprocess.DEVNULL)
                samples[name].append(time.perf_counter() - start)
    return {name: {'count': len(vals), 'median_ms': 1000*median(vals)} for name, vals in samples.items()}


def main():
    lengths = defaultdict(list)
    prepare = unidoc_fast.prepare_pdb
    run = unidoc_fast.subprocess.run
    def timed(fn, name, *args, **kwargs):
        start = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            lengths[name].append(time.perf_counter()-start)
    def prepare_timed(*args, **kwargs):
        return timed(prepare, 'prepare_pdb_and_sequence', *args, **kwargs)
    def run_timed(command, **kwargs):
        return timed(run, Path(command[0]).name, command, **kwargs)

    output = ROOT/'profile-fast'
    output.mkdir()
    inputs = sorted((BASE/'benchmark/holdout').glob('*.pdb'), key=lambda p: (-p.stat().st_size, p.name))
    inherited = BASE/'benchmark/holdout-reference/chopping_merizo.txt'
    with patch.object(unidoc_fast,'prepare_pdb',prepare_timed), patch.object(unidoc_fast.subprocess,'run',run_timed):
        start = time.perf_counter()
        with (output/'run.log').open('w') as log:
            unidoc_fast.run_batch(inputs, inherited, output/'raw.tsv', 4, os.environ.copy(), log)
        wall = time.perf_counter()-start
    report = {'inputs': len(inputs), 'wall_seconds': wall,
              'summed_worker_calls': {key:{'count':len(vals),'seconds':sum(vals)} for key, vals in lengths.items()}}
    report['tiny_input_command_probe'] = startup_probe()
    (output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
