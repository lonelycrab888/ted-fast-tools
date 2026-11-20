"""Retry legacy UniDoc failures after the PDB header compatibility correction."""
import os
import json
from pathlib import Path
import sys
from run_supplement import BASE, METHODS, dump, read_tsv, run_consensus, run_method, write_lines

items = {row['afdb_model_id']: {**row, 'length': int(row['length'])}
         for row in read_tsv(BASE / 'provenance/inputs.tsv')}
env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
           OPENBLAS_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
total = recovered = 0
for path in sorted((BASE / 'logs').glob('gpu*/batch[0-9][0-9][0-9]/result.json')):
    batch = path.parent
    original = json.loads(path.read_text())
    rows = original['structures']
    selected = {m: items[m] for m, row in rows.items()
                if any(e.startswith('unidoc:') and 'exit status 134' in e for e in row['errors'])}
    if not selected:
        continue
    stage = batch / 'unidoc-header-retry'
    if stage.exists():
        continue
    stage.mkdir()
    original_file = stage / 'original-result.json'
    dump(original_file, original)
    chop = {}
    for method in METHODS:
        chop[method] = {line.split('\t')[0]: line for line in
                        (batch / f'chopping_{method}.txt').read_text().splitlines()}
    found, failures, attempts = run_method('unidoc', selected, stage, env, chop['merizo'])
    success = {m: item for m, item in selected.items() if m in found and m not in failures}
    chop['unidoc'].update({m: found[m] for m in success})
    eligible = {m: item for m, item in success.items() if all(m in chop[x] for x in METHODS)}
    consensus, consensus_errors = run_consensus(eligible, chop, stage, env)
    for model in selected:
        details = rows[model]
        if model in success:
            details['methods']['unidoc'] = found[model].split('\t')[4]
            details['errors'] = [e for e in details['errors'] if not e.startswith('unidoc:')]
            if model in consensus:
                details['consensus'] = consensus[model]
            elif model in consensus_errors:
                details['errors'].append(consensus_errors[model])
                details['consensus'] = None
        elif model in failures:
            details['errors'].append(failures[model])
    original['attempts'].extend(attempts)
    write_lines(batch / 'chopping_unidoc.txt', chop['unidoc'])
    write_lines(batch / 'consensus.tsv', {m: r['consensus'] for m, r in rows.items() if r['consensus']})
    dump(path, original)
    total += len(selected)
    recovered += len(success)
    print(f'{path.parent}: recovered {len(success)}/{len(selected)} UniDoc rows',flush=True)
print(f'UniDoc retry total {recovered}/{total}',flush=True)
