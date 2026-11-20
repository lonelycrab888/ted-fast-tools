"""Run TED consensus on the frozen set of unmatched plddt70_v1 structures."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

imported = Path(__file__).resolve().parents[2] / 'plddt70_v1_ted_v4'
sys.path.insert(0, str(imported))
from split_ted_v4 import read_parent_pdb, write_domain

BASE = Path(__file__).resolve().parents[1]
BENCH = BASE.parents[1]
TED = Path('/home/zhr/tools/ted-tools/ted_consensus_1.0')
METHODS = ('merizo', 'unidoc', 'chainsaw')
GPUS = (6, 7)
BATCH_SIZE = 32


def read_tsv(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle, delimiter='\t'))


def write_tsv(path, fields, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fields, delimiter='\t', lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def dump(path, obj):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def read_rows(path, expected, width):
    out = {}
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        cols = line.split('\t')
        if len(cols) != width or cols[0] not in expected or cols[0] in out:
            raise ValueError(f'Malformed, unexpected or duplicate row in {path}: {line}')
        if int(cols[2]) != expected[cols[0]]['length']:
            raise ValueError(f'Residue count mismatch in {path}: {line}')
        out[cols[0]] = line
    return out


def run_logged(cmd, cwd, env, log_path):
    start = time.monotonic()
    with log_path.open('w') as log:
        proc = subprocess.run(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
    return proc.returncode, round(time.monotonic() - start, 2)


def write_lines(path, rows):
    path.write_text(''.join(f'{row}\n' for row in rows.values()))


def run_method(method, selected, parent, env, inherited, depth='initial'):
    if not selected:
        return {}, {}, []
    work = parent / f'{method}-{depth}'
    work.mkdir()
    inputs = work / 'inputs'
    inputs.mkdir()
    outputs = work / 'outputs'
    outputs.mkdir()
    for model, item in selected.items():
        shutil.copy2(item['parent_pdb'], inputs / f'{model}.pdb')
    cmd = ['bash', str(TED / 'scripts/segment.sh'), '-i', str(inputs), '-o', str(outputs), '-m', method]
    if method == 'unidoc':
        file = work / 'merizo.tsv'
        write_lines(file, {m: inherited[m] for m in selected})
        cmd.extend(('-c', str(file)))
    log = work / 'run.log'
    code, seconds = run_logged(cmd, TED, env, log)
    result_file = outputs / f'chopping_{method}.txt'
    problem = f'exit={code}'
    found = {}
    if code == 0:
        try:
            found = read_rows(result_file, selected, 6)
        except (OSError, ValueError) as error:
            problem = str(error)
    log_text = log.read_text(errors='replace')
    native_errors = {}
    for line in log_text.splitlines():
        if line.startswith('UNIDOC_ERROR\t'):
            _, model, detail = line.split('\t', 2)
            native_errors[model] = detail
    missing = {m: item for m, item in selected.items() if m not in found}
    attempts = [{'method': method, 'attempt': depth, 'targets': len(selected),
                 'rows': len(found), 'exit_code': code, 'seconds': seconds,
                 'log': str(log.relative_to(BASE))}]
    failures = {m: f'{method}: {problem}; see {log.relative_to(BASE)}'
                for m in missing}
    if missing and (len(selected) > 1 or depth == 'initial'):
        parts = list(missing.items())
        size = max(1, (len(parts) + 1) // 2)
        for index, offset in enumerate(range(0, len(parts), size)):
            subset = dict(parts[offset:offset + size])
            rows, retry_errors, retry_attempts = run_method(method, subset, parent, env, inherited, f'{depth}-{index}')
            found.update(rows)
            attempts.extend(retry_attempts)
            failures.update(retry_errors)
            for model in rows:
                if model not in retry_errors:
                    failures.pop(model, None)
    failures.update({m: f'unidoc: {error}; see {log.relative_to(BASE)}'
                     for m, error in native_errors.items() if m in found and m not in missing})
    # Large PDB inputs are copied per attempt solely for TED's file-local temporary outputs.
    shutil.rmtree(inputs)
    return found, failures, attempts


def run_consensus(selected, chop, parent, env, label='initial'):
    if not selected:
        return {}, {}
    work = parent / f'consensus-{label}'
    work.mkdir()
    for method in METHODS:
        write_lines(work / f'chopping_{method}.txt', {m: chop[method][m] for m in selected})
    log = work / 'run.log'
    common = ['chopping_merizo.txt', 'chopping_chainsaw.txt', 'chopping_unidoc.txt']
    cmd = [sys.executable, str(TED / 'scripts/get_consensus.py'), '-c', *common, '-o', 'raw.tsv']
    code, _ = run_logged(cmd, work, env, log)
    result = work / 'consensus.tsv'
    if code == 0:
        cmd = [sys.executable, str(TED / 'scripts/filter_domains_consensus.py'), 'raw.tsv', '-o', result.name]
        code, _ = run_logged(cmd, work, env, work / 'filter.log')
    found = {}
    if code == 0:
        try:
            found = read_rows(result, selected, 9)
        except (OSError, ValueError):
            pass
    missing = {m: item for m, item in selected.items() if m not in found}
    if not missing:
        return found, {}
    if len(selected) == 1 and label != 'initial':
        return found, {m: f'consensus: missing or invalid output; see {log.relative_to(BASE)}' for m in missing}
    errors = {}
    parts = list(missing.items())
    size = max(1, (len(parts) + 1) // 2)
    for index, offset in enumerate(range(0, len(parts), size)):
        rows, failures = run_consensus(dict(parts[offset:offset + size]), chop, parent, env, f'{label}-{index}')
        found.update(rows)
        errors.update(failures)
    return found, errors


def run_lane(gpu, gpu_uuid, items):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_uuid, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
    probe = 'import torch; assert torch.cuda.device_count() == 1; print(torch.cuda.get_device_name(0))'
    subprocess.run([sys.executable, '-c', probe], env=env, check=True)
    lane = BASE / 'logs' / f'gpu{gpu}'
    lane.mkdir(exist_ok=True)
    start = time.monotonic()
    for offset in range(0, len(items), BATCH_SIZE):
        selected = dict(items[offset:offset + BATCH_SIZE])
        index = offset // BATCH_SIZE
        batch = lane / f'batch{index:03d}'
        done = batch / 'result.json'
        if done.exists():
            continue
        if batch.exists():
            pending = lane / f'batch{index:03d}-interrupted-{int(time.time())}'
            batch.rename(pending)
        batch.mkdir()
        chop = {}
        errors = {m: [] for m in selected}
        attempts = []
        for method in METHODS:
            method_targets = selected
            if method == 'unidoc':
                method_targets = {m: selected[m] for m in chop['merizo']}
            found, failures, method_attempts = run_method(method, method_targets, batch, env, chop.get('merizo', {}))
            chop[method] = found
            for model, error in failures.items():
                errors[model].append(error)
            attempts.extend(method_attempts)
            write_lines(batch / f'chopping_{method}.txt', found)
        shared = selected.keys() & chop['merizo'].keys() & chop['unidoc'].keys() & chop['chainsaw'].keys()
        ready = {m: item for m, item in selected.items() if m in shared}
        consensus, failures = run_consensus(ready, chop, batch, env)
        for model, error in failures.items():
            errors[model].append(error)
        write_lines(batch / 'consensus.tsv', consensus)
        rows = {}
        for model in selected:
            fields = {method: chop[method][model].split('\t')[4] if model in chop[method] else 'MISSING'
                      for method in METHODS}
            rows[model] = {'consensus': consensus.get(model), 'methods': fields, 'errors': errors[model]}
        dump(done, {'gpu': gpu, 'batch': index, 'structures': rows, 'attempts': attempts})
        print(f'GPU {gpu} batch {index:03d}: {min(offset + BATCH_SIZE, len(items))}/{len(items)} processed, '
              f'{len(consensus)}/{len(selected)} consensus rows; lane elapsed {(time.monotonic() - start) / 60:.1f} min',
              flush=True)


def export(items, lanes):
    result = {}
    for gpu in GPUS:
        for path in sorted((BASE / 'logs' / f'gpu{gpu}').glob('batch[0-9][0-9][0-9]/result.json')):
            rows = json.loads(path.read_text())['structures']
            assert not (result.keys() & rows.keys()), path
            result.update(rows)
    assert result.keys() == items.keys(), (len(result), len(items))
    identity = ['structure_id', 'accession', 'afdb_model_id', 'fragment']
    structure_fields = identity + ['residues', 'status', 'high', 'medium', 'merizo', 'unidoc', 'chainsaw', 'reason']
    domain_fields = ['domain_id', 'structure_id', 'accession', 'afdb_model_id', 'boundaries', 'consensus',
                     'residues', 'atoms', 'mean_ca_plddt', 'pdb_path', 'pdb_sha256']
    domain_dir = BASE / 'domains'
    domain_dir.mkdir(exist_ok=True)
    rows = []
    domains = []
    for model, item in items.items():
        raw = result[model]
        row = {field: item[field] for field in identity}
        methods = {name: raw['methods'][name] for name in METHODS}
        row.update({name: value if value in {'MISSING', 'NULL', 'NO_SS'} else 'ok' for name, value in methods.items()})
        row.update(residues=item['length'], high=0, medium=0)
        problems = list(raw['errors'])
        line = raw['consensus']
        if line is None:
            row['status'] = 'failed'
            row['reason'] = ' | '.join(problems) or 'missing consensus'
            rows.append(row)
            continue
        status = 'no_domain'
        accepted = []
        parsed = line.split('\t')
        for level, count, value in zip(('high', 'medium'), parsed[3:5], parsed[6:8]):
            boundaries = [] if value == 'na' else value.split(',')
            if len(boundaries) != int(count):
                problems.append(f'{level}: reported domain count does not match bounds')
                status = 'failed'
                continue
            for boundary in boundaries:
                try:
                    segments = [tuple(map(int, seg.split('-'))) for seg in boundary.split('_')]
                    positions = [i for a,b in segments for i in range(a, b + 1)]
                    if (not positions or positions != sorted(set(positions)) or min(positions) < 1
                            or max(positions) > item['length'] or any(b - a + 1 < 5 for a,b in segments)
                            or len(positions) < 25):
                        raise ValueError(f'invalid TED {level} boundary {boundary}')
                    accepted.append((level, boundary, positions))
                except ValueError as error:
                    problems.append(str(error))
                    status = 'failed'
        if accepted:
            try:
                atoms, ca = read_parent_pdb(Path(item['parent_pdb']))
                assert len(ca) == item['length']
                for index, (level, boundary, positions) in enumerate(accepted, 1):
                    domain_id = f'{model}_TES{index:02d}'
                    rel = Path('domains') / f'{domain_id}.pdb'
                    path = BASE / rel
                    if path.exists():
                        # Re-export a resumed run through a temporary PDB and compare bytes.
                        temporary = path.with_suffix('.verify')
                        natoms, hash_value = write_domain(temporary, positions, atoms)
                        if digest(path) != hash_value:
                            raise ValueError(f'Existing domain differs from validated output: {path}')
                        temporary.unlink()
                    else:
                        natoms, hash_value = write_domain(path, positions, atoms)
                    domains.append(dict(domain_id=domain_id, structure_id=item['structure_id'], accession=item['accession'],
                                        afdb_model_id=model, boundaries=boundary, consensus=level, residues=len(positions),
                                        atoms=natoms, mean_ca_plddt=f'{sum(ca[i] for i in positions)/len(positions):.4f}',
                                        pdb_path=rel.as_posix(), pdb_sha256=hash_value))
                    row[level] += 1
                if status != 'failed':
                    status = 'with_domain'
            except (AssertionError, OSError, ValueError) as error:
                status = 'failed'
                problems.append(f'PDB export: {error}')
        row['status'] = status
        row['reason'] = ' | '.join(problems) or ('no high/medium consensus' if status == 'no_domain' else '')
        rows.append(row)
    failures = [r for r in rows if r['status'] == 'failed']
    empty = [r for r in rows if r['status'] == 'no_domain']
    method_errors = [r for r in rows if result[r['afdb_model_id']]['errors']]
    write_tsv(BASE / 'domains.tsv', domain_fields, domains)
    write_tsv(BASE / 'structures.tsv', structure_fields, rows)
    write_tsv(BASE / 'failed.tsv', structure_fields, failures)
    write_tsv(BASE / 'no_domain.tsv', structure_fields, empty)
    write_tsv(BASE / 'method_errors.tsv', structure_fields, method_errors)
    retained = {r['afdb_model_id'] for r in domains}
    paths = {str(p.relative_to(BASE)) for p in domain_dir.glob('*.pdb')}
    assert paths == {r['pdb_path'] for r in domains} and len(paths) == len(domains)
    assert all(digest(BASE / r['pdb_path']) == r['pdb_sha256'] for r in domains)
    assert sum(r['high'] + r['medium'] for r in rows) == len(domains)
    summary = {
        'input_structures': len(items), 'gpu_split': {gpu: len(lanes[gpu]) for gpu in GPUS},
        'structures_with_domains': len(retained), 'structures_without_domains': len(empty),
        'failed_structures': len(failures), 'structures_with_method_errors': len(method_errors),
        'structures_with_method_errors_and_domains': sum(r['afdb_model_id'] in retained for r in method_errors),
        'domains': dict(Counter(r['consensus'] for r in domains)),
        'by_fragment': dict(Counter(f"{'F1' if r['fragment'] == '1' else 'F2+'}:{r['status']}" for r in rows)),
        'exported_pdbs_verified': len(paths),
    }
    dump(BASE / 'summary.json', summary)
    dump(BASE / 'validation.json', {'passed': True, 'criteria': 'Every input linked to exactly one status, all exported domain PDB and manifests verified.',
         'summary': summary, 'sha256': {f: digest(BASE / f) for f in
                                       ('domains.tsv', 'structures.tsv', 'failed.tsv', 'no_domain.tsv', 'method_errors.tsv', 'summary.json')}})
    return summary


def main():
    data = BENCH.parents[1]
    source_path = data / 'HUMAN/manifests/human_gallery.json'
    source = {r['structure_id']: r for r in json.loads(source_path.read_text())}
    old = read_tsv(imported / 'unmatched.tsv')
    gallery = {r['structure_id'] for r in json.loads((BENCH / 'datasets/plddt70_v1/gallery.json').read_text())}
    items = {}
    for row in old:
        sid, model = row['structure_id'], row['afdb_model_id']
        entry = source[sid]
        path = Path(entry['path'])
        assert sid in gallery and sid.removeprefix('hu_') == model == path.stem and model not in items
        assert row['accession'] == entry['accession']
        with path.open('rb') as handle:
            length = sum(l.startswith(b'ATOM  ') and l[12:16].strip() == b'CA' for l in handle)
        assert length > 0
        items[model] = {**row, 'length': length, 'parent_pdb': str(path)}
    assert len(items) == 2349
    uuid_list = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    gpu_uuids = {int(idx): uuid for idx, uuid in (line.split(', ') for line in uuid_list.splitlines())}
    lanes = {gpu: [] for gpu in GPUS}
    loads = dict.fromkeys(GPUS, 0)
    # Greedy length-weighted split balances long targets between physical GPUs.
    for model, item in sorted(items.items(), key=lambda pair: (-pair[1]['length'], pair[0])):
        gpu = min(GPUS, key=loads.__getitem__)
        lanes[gpu].append((model, item))
        loads[gpu] += 1_000_000 + item['length'] ** 2
    plan = [dict(item, gpu=gpu) for gpu, entries in lanes.items() for _, item in entries]
    write_tsv(BASE / 'provenance/inputs.tsv', list(plan[0]), plan)
    dump(BASE / 'provenance/run.json', dict(started=datetime.now().astimezone().isoformat(),
         structures=len(items), gpu_uuids={gpu: gpu_uuids[gpu] for gpu in GPUS}, batch_size=BATCH_SIZE,
         unmatched_sha256=digest(imported / 'unmatched.tsv'), source_manifest_sha256=digest(source_path),
         ted=str(TED), python=sys.executable))
    dump(BASE / 'status.json', {'state': 'running', 'structures': len(items), 'pid': os.getpid()})
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=len(GPUS)) as executor:
        jobs = [executor.submit(run_lane, gpu, gpu_uuids[gpu], lanes[gpu]) for gpu in GPUS]
        for job in jobs:
            job.result()
    summary = export(items, lanes)
    dump(BASE / 'status.json', {'state': 'complete', 'wall_seconds': round(time.monotonic() - start, 2), **summary})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        dump(BASE / 'status.json', {'state': 'failed', 'error': traceback.format_exc()})
        raise
