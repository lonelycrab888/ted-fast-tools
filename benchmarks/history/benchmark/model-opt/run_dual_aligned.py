"""Run the two-GPU TED_fast workflow with the original supplement's exact GPU assignment."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[4]
ROOT = REPO.parent / "TED_fast"
sys.path.insert(0, str(REPO))
import run_fast

BENCH = ROOT / 'benchmark'
OLD = ROOT.parent / 'BenchMark/HumanMotifbench/datasets/plddt70_v1_tes_supplement'
CARDS = (6, 7)


def read_json(path):
    return json.loads(path.read_text())


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def utc():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    output = parser.parse_args().output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    run = output / 'run'
    inputs = BENCH / 'full-input'
    old = read_json(OLD / 'status.json')
    reference = read_json(BENCH / 'full-reference/timing.json')
    provenance = read_json(OLD / 'provenance/run.json')
    assert old['state'] == 'complete' and old['wall_seconds'] == reference['wall_seconds']
    source = OLD / 'provenance/inputs.tsv'
    with source.open() as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    paths = {p.stem: p for p in inputs.glob('*.pdb')}
    assert len(rows) == len(paths) == 2349
    assert set(paths) == {r['afdb_model_id'] for r in rows}
    assert all(paths[r['afdb_model_id']].resolve() == Path(r['parent_pdb']).resolve()
               and paths[r['afdb_model_id']].is_file() for r in rows)
    assignments = [[r['afdb_model_id'] for r in rows if int(r['gpu']) == gpu] for gpu in CARDS]
    assert [len(s) for s in assignments] == [old['gpu_split'][str(i)] for i in CARDS]
    ids = run_fast.gpu_ids(CARDS)
    assert ids == [provenance['gpu_uuids'][str(i)] for i in CARDS]

    def archived_split(pdbs, count):
        assert count == 2 and {p.stem for p in pdbs} == set(paths)
        by_id = {p.stem: p for p in pdbs}
        return [[by_id[model] for model in shard] for shard in assignments]

    argv = ['run_fast.py', '-i', str(inputs), '-o', str(run), '--gpus', '6', '7',
            '--variant', 'ted-fast', '--structures-per-gpu', '1', '--overlap',
            '--unidoc-backend', 'fast', '--unidoc-batch-size', '32', '--unidoc-workers', '4']
    config = {'start_utc': utc(), 'fast_command_args': argv, 'gpu_uuids': dict(zip(CARDS, ids)),
              'structures_per_gpu': dict(zip(CARDS, map(len, assignments))),
              'input_manifest': str(source), 'manifest_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'historical_status': str(OLD/'status.json'), 'historical_seconds': old['wall_seconds']}
    save_json(output/'config.json', config)
    print('Starting aligned full inference', config, flush=True)
    start = time.perf_counter()
    # Keep the installed multi-GPU scheduler, workers, model filtering and merger.
    # Replace only its initial file-size split with the original per-GPU manifest.
    with patch.object(run_fast, 'split_paths', archived_split), patch.object(sys, 'argv', argv):
        run_fast.main()
    at_consensus = time.perf_counter()
    print('Consensus written; exporting and validating domain PDBs', utc(), flush=True)
    verify_cmd = [sys.executable, str(BENCH/'verify_full.py'), '--run', str(run),
                  '--report', str(output/'export-validation.json'),
                  '--comparison', str(output/'table-comparison.json')]
    with (output/'export.log').open('w') as log:
        subprocess.run(verify_cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    finished = time.perf_counter()
    verify = read_json(output/'export-validation.json')
    # The archived four-card verifier uses a four-card label irrespective of GPU count.
    # Correct that label in this run's report without altering the historical verifier.
    speedup_by_stage_sum = verify.pop('speedup_old_dual_to_new_four')
    verify['speedup_vs_historical_dual_by_stage_sum'] = speedup_by_stage_sum
    save_json(output/'export-validation.json', verify)
    tables = read_json(output/'table-comparison.json')
    timing = read_json(run/'timing.json')
    assert verify['passed'] and set(timing['gpus']) == {'6','7'} and timing['variant'] == 'ted-fast'
    assert all(m['rows_identical'] == m['rows'] == len(rows) for m in tables.values())
    assert verify['domain_pdbs_identical'] == verify['reference_domain_pdbs'] == 13258
    for gpu, shard in zip(CARDS, assignments):
        actual = [Path(p).stem for p in (run/f'gpu{gpu}/merizo/inputs.txt').read_text().splitlines()]
        assert actual == shard, f'GPU {gpu} did not get the historical input list and order'
    elapsed = finished - start
    baseline = old['wall_seconds']
    report = {'inputs': len(rows), 'same_order_and_models_per_gpu': True, 'gpu_uuids': timing['gpus'],
              'structures_per_gpu': config['structures_per_gpu'], 'configuration': argv,
              'baseline_source': str(OLD/'status.json'), 'original_wall_seconds': baseline,
              'new_inference_through_consensus_seconds': at_consensus-start,
              'new_export_and_verification_seconds': finished-at_consensus,
              'new_end_to_end_wall_seconds': elapsed, 'same_gpu_end_to_end_speedup': baseline/elapsed,
              'seconds_saved': baseline-elapsed, 'percent_wall_time_reduction': 100*(1-elapsed/baseline),
              'tenfold_threshold_seconds': baseline/10, 'tenfold_reached': elapsed <= baseline/10,
              'pipeline_internal_seconds': timing['wall_seconds'],
              'export_validation_internal_seconds': verify['fast_export_and_verify_seconds'],
              'historical_benchmark_stage_sum_seconds': verify['fast_time_seconds_with_export'],
              'stage_sum_speedup': speedup_by_stage_sum, 'domain_counts': verify['counts'],
              'identical_domain_pdb_sha256': verify['domain_pdbs_identical'],
              'identical_table_rows': {k:v['rows_identical'] for k,v in tables.items()},
              'finished_utc': utc()}
    save_json(output/'summary.json', report)
    print('Verified measured result', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
