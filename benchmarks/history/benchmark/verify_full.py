"""Validate the full TED_fast benchmark and export retained domains locally."""
import argparse
from collections import Counter
import csv
from hashlib import sha256
import json
from pathlib import Path
import sys
import time

from compare import compare, read

BASE = Path(__file__).resolve().parent
BENCH = BASE.parents[1]/'BenchMark/HumanMotifbench'
DATA = BENCH/'datasets'
OFFICIAL = DATA/'plddt70_v1_ted_v4'
OLD = DATA/'plddt70_v1_tes_supplement'
sys.path.insert(0, str(OFFICIAL))
from split_ted_v4 import read_parent_pdb, write_domain


def sha(path):
    return sha256(path.read_bytes()).hexdigest()


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run', type=Path, default=BASE/'full-fast')
parser.add_argument('--report', type=Path, default=BASE/'full-validation.json')
parser.add_argument('--comparison', type=Path, default=BASE/'full-comparison.json')
args = parser.parse_args()
FAST = args.run.resolve()

start = time.perf_counter()
comparison = compare(BASE/'full-reference', FAST)
timing = json.loads((FAST/'timing.json').read_text())
reference = json.loads((BASE/'full-reference/timing.json').read_text())
structures = read(FAST/'consensus.tsv')
args.comparison.write_text(json.dumps(comparison, indent=2)+'\n')
assert len(structures) == timing['inputs'] == reference['input_structures'] == 2349
with (OLD/'provenance/inputs.tsv').open() as f:
    parents = {r['afdb_model_id']:r for r in csv.DictReader(f, delimiter='\t')}
assert parents.keys() == structures.keys()

counts = Counter()
manifest = []
exports = FAST/'domains'
exports.mkdir(exist_ok=True)
for model, row in structures.items():
    domains = []
    for level, n, ranges in zip(('high', 'medium'), row[2:4], row[5:7]):
        boundaries = [] if ranges == 'na' else ranges.split(',')
        assert len(boundaries) == int(n)
        domains.extend((level, bounds) for bounds in boundaries)
    counts['with_domain' if domains else 'no_domain'] += 1
    if not domains:
        continue
    atoms, ca = read_parent_pdb(Path(parents[model]['parent_pdb']))
    assert len(ca) == int(row[1])
    for i, (level, bounds) in enumerate(domains, 1):
        segments = [tuple(map(int, s.split('-'))) for s in bounds.split('_')]
        positions = [p for lo, hi in segments for p in range(lo, hi+1)]
        assert len(positions) >= 25 and positions == sorted(set(positions))
        assert all(hi-lo+1 >= 5 for lo, hi in segments)
        path = exports/f'{model}_TES{i:02d}.pdb'
        if path.exists():
            temp = path.with_suffix('.verify')
            natoms, value = write_domain(temp, positions, atoms)
            assert value == sha(path)
            temp.unlink()
        else:
            natoms, value = write_domain(path, positions, atoms)
        counts[level] += 1
        manifest.append([path.stem, parents[model]['structure_id'], model, level, bounds, len(positions), natoms,
                         value, str(path.relative_to(FAST))])
with (FAST/'domains.tsv').open('w', newline='') as f:
    w = csv.writer(f, delimiter='\t', lineterminator='\n')
    w.writerow(['domain_id', 'structure_id', 'afdb_model_id', 'consensus', 'boundaries', 'residues', 'atoms', 'pdb_sha256', 'pdb_path'])
    w.writerows(manifest)
assert len(list(exports.glob('*.pdb'))) == len(manifest)
with (OLD/'domains.tsv').open() as f:
    previous = {r['domain_id']: r['pdb_sha256'] for r in csv.DictReader(f, delimiter='\t')}
identical = sum(previous.get(r[0]) == r[7] for r in manifest)
validation_seconds = time.perf_counter()-start
total = timing['wall_seconds'] + validation_seconds
result = {'inputs':len(structures), 'counts':dict(counts), 'old_gpus':list(reference['gpu_split']),
          'fast_gpus':list(timing['gpus']), 'old_time_seconds_with_export':reference['wall_seconds'],
          'fast_time_seconds_consensus':timing['wall_seconds'], 'fast_export_and_verify_seconds':validation_seconds,
          'fast_time_seconds_with_export':total, 'speedup_old_dual_to_new_four':reference['wall_seconds']/total,
          'domain_pdbs_identical':identical, 'reference_domain_pdbs':len(previous),
          'exact_consensus_rows':comparison['consensus.tsv']['rows_identical'],
          'retained_boundaries_identical':comparison['consensus.tsv']['geometry_identical'],
          'stages':timing['stages_seconds'], 'passed': all(x['rows_identical']==2349 for x in comparison.values())
          and identical==len(manifest)==len(previous)}
args.report.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result, indent=2))
if not result['passed']:
    sys.exit(1)
