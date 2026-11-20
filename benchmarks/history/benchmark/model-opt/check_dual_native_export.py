"""Recheck a dual-GPU run with the exact exporter and validation code used by the historical TED job."""
import argparse
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT.parent / 'BenchMark/HumanMotifbench/datasets/plddt70_v1_tes_supplement'
sys.path.insert(0, str(OLD/'provenance'))
import run_supplement as old_export


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('benchmark', type=Path)
    args = parser.parse_args()
    output = args.benchmark.resolve()
    run = output/'run'
    wall = json.loads((output/'summary.json').read_text())
    manifest = old_export.read_tsv(OLD/'provenance/inputs.tsv')
    by_id = {r['afdb_model_id']:{**r, 'length':int(r['length'])} for r in manifest}
    original_order = old_export.read_tsv(old_export.imported/'unmatched.tsv')
    items = {r['afdb_model_id']:by_id[r['afdb_model_id']] for r in original_order}
    assert items.keys()==by_id.keys() and len(items)==2349
    lanes = {gpu:[(r['afdb_model_id'],items[r['afdb_model_id']]) for r in manifest if int(r['gpu'])==gpu]
             for gpu in old_export.GPUS}
    method_errors = {}
    for path in run.glob('gpu*/unidoc-batches/batch*/unidoc/run.log'):
        for line in path.read_text().splitlines():
            if line.startswith('UNIDOC_ERROR\t'):
                _, name, error = line.split('\t',2)
                method_errors.setdefault(name,[]).append(error)
    expected = json.loads((OLD/'validation.json').read_text())
    with TemporaryDirectory(prefix='native-export-', dir=output) as folder:
        work = Path(folder)
        start = time.perf_counter()
        choppings = {m:old_export.read_rows(run/f'chopping_{m}.txt', items, 6) for m in old_export.METHODS}
        consensus = old_export.read_rows(run/'consensus.tsv', items, 9)
        assert len(consensus)==2349 and all(len(d)==2349 for d in choppings.values())
        for gpu, ordered in lanes.items():
            for offset in range(0,len(ordered),old_export.BATCH_SIZE):
                batch = work/'logs'/f'gpu{gpu}'/f'batch{offset//old_export.BATCH_SIZE:03d}'
                batch.mkdir(parents=True)
                rows = {m:{'consensus':consensus[m], 'methods':{k:v[m].split('\t')[4] for k,v in choppings.items()},
                           'errors':method_errors.get(m,[])} for m, _ in ordered[offset:offset+old_export.BATCH_SIZE]}
                old_export.dump(batch/'result.json',{'structures':rows})
        export_start = time.perf_counter()
        with patch.object(old_export, 'BASE', work):
            native_summary = old_export.export(items, lanes)
        end = time.perf_counter()
        native_validation = json.loads((work/'validation.json').read_text())
        assert native_validation['passed'] and native_validation['summary'] == expected['summary']
        identical_files = {name:old_export.digest(work/name)==digest for name,digest in expected['sha256'].items()}
        assert all(identical_files.values()), identical_files
        # Retain the exact production-format tables. Their PDB paths point to the
        # lasting, independently hash-checked exports through this local symlink.
        retained = output/'production-manifests'
        retained.mkdir()
        for name in (*expected['sha256'], 'validation.json'):
            shutil.copy2(work/name, retained/name)
        (retained/'domains').symlink_to('../run/domains', target_is_directory=True)
    combined = wall['new_inference_through_consensus_seconds'] + end-start
    report = {'historical_export_source': str(OLD/'provenance/run_supplement.py'),
              'historical_export_source_sha256':old_export.digest(OLD/'provenance/run_supplement.py'),
              'input_adapter_seconds':export_start-start, 'original_export_and_validation_seconds':end-export_start,
              'full_postprocessing_stage_seconds':end-start, 'same_export_stage_sum_seconds':combined,
              'same_export_stage_sum_speedup':wall['original_wall_seconds']/combined,
              'same_export_stage_sum_tenfold':combined <= wall['tenfold_threshold_seconds'],
              'identical_original_summary_and_validation':native_validation==expected,
              'identical_manifest_sha256':identical_files, 'exported_pdbs_verified':native_summary['exported_pdbs_verified'],
              'failed_structures':native_summary['failed_structures'], 'structures_with_method_errors':native_summary['structures_with_method_errors']}
    path = output/'native-export-check.json'
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2), flush=True)


if __name__ == '__main__':
    main()
