"""Run the pinned three-method TED workflow on a directory of AFDB PDBs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from threading import BoundedSemaphore
import time

BASE = Path(__file__).resolve().parent
CONSENSUS = BASE / 'consensus'
METHODS = ('merizo', 'chainsaw', 'unidoc')
OUTPUTS = (*(f'chopping_{name}.txt' for name in METHODS), 'consensus.tsv')


def launch(cmd, env, directory, log):
    start = time.perf_counter()
    with log.open('w') as f:
        subprocess.run(cmd, cwd=directory, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
    return time.perf_counter() - start


def method(name, paths, output, env, *, profile=False, workers=1, variant="ted-fast",
           gpu_slots=None, inherited=None, whole_chain=False, packed=False, unidoc_backend="fast"):
    output.mkdir()
    inner = output/'pdb'
    fast_unidoc = name == 'unidoc' and unidoc_backend == 'fast'
    if name == 'chainsaw' or (name == 'unidoc' and not fast_unidoc):
        inner.mkdir()
        for path in paths:
            shutil.copy2(path, inner/path.name)
        paths = [inner/p.name for p in paths]
    targets = output/'inputs.txt'
    targets.write_text(''.join(f'{p}\n' for p in paths))
    raw = output/'raw.tsv'
    if name == 'unidoc':
        if not whole_chain:
            inherited = inherited or output.parent/'chopping_merizo.txt'
        if fast_unidoc:
            from modules.unidoc.unidoc_fast import run_batch
            start = time.perf_counter()
            with (output/'run.log').open('w') as log:
                run_batch(paths, inherited, raw, workers, env, log)
            elapsed = time.perf_counter() - start
            return elapsed + filter_method(name, raw, paths, env)
        wrapper = BASE/'modules/unidoc/upstream/Run_UniDoc_from_scratch_structure_afdb.py'
        if workers == 1:
            command = [sys.executable, wrapper, '-l', targets, '--out', raw, '--inherit_chopping', inherited]
        else:
            command = [sys.executable, BASE/'modules/unidoc/run_unidoc.py', targets, inherited, raw, '--workers', str(workers)]
    else:
        command = [sys.executable, BASE/'run_method.py', name, '--variant', variant]
        if profile:
            command.extend(('--profile', output/'profile.pstat'))
        if name == 'merizo':
            if packed:
                command.append('--packed-confidence')
            command.extend(('-l', targets, '--out', raw))
        else:
            command.extend(('--structure_directory', inner, '-o', raw, '--append'))
    try:
        with (gpu_slots if gpu_slots is not None else nullcontext()):
            elapsed = launch(command, env, output, output/'run.log')
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f'{name} failed in {output}. See {output / "run.log"}') from error
    elapsed += filter_method(name, raw, paths, env)
    if inner.exists():
        shutil.rmtree(inner)
    return elapsed


def filter_method(name, raw, paths, env):
    output = raw.parent
    chop = output.parent/f'chopping_{name}.txt'
    command = [sys.executable, CONSENSUS/'filter_domains.py', raw, '-o', chop,
               '--offset_resi', str(int(name == 'chainsaw'))]
    elapsed = launch(command, env, output, output/'filter.log')
    check_rows(chop, {p.stem for p in paths}, 6)
    return elapsed


def check_rows(path, expected, width):
    rows = [line.split('\t') for line in path.read_text().splitlines()]
    if len(rows) != len(expected) or {r[0] for r in rows} != expected or any(len(r) != width for r in rows):
        raise ValueError(f'Missing, duplicate or malformed target rows in {path}')


def stream_unidoc(merizo, paths, output, env, workers, cpu_pool, batch_size, unidoc_backend='fast'):
    """Hand off complete Merizo rows while the GPU process is still running."""
    root = output/'unidoc-batches'
    root.mkdir()
    raw = output/'merizo/raw.tsv'
    by_id = {path.stem: path for path in paths}
    cursor = 0
    pending = []
    submitted = []
    seen = set()
    active = None
    uni_seconds = 0.0
    lane_start = time.perf_counter()
    first_submit = None
    merizo_seconds = None

    while True:
        now_done = merizo.done()
        if now_done and merizo_seconds is None:
            merizo_seconds = merizo.result()
        if raw.is_file():
            with raw.open('rb') as f:
                f.seek(cursor)
                fresh = f.read()
            complete = fresh.rfind(b'\n') + 1
            cursor += complete
            for line in fresh[:complete].decode().splitlines():
                model = line.split('\t', 1)[0]
                if model not in by_id or model in seen:
                    raise ValueError(f'Unexpected or duplicate Merizo row for {model} in {raw}')
                seen.add(model)
                pending.append((by_id[model], line))
        if active is not None and active.done():
            uni_seconds += active.result()
            active = None
        if active is None and pending and (len(pending) >= batch_size or now_done):
            batch = root/f'batch{len(submitted):04d}'
            batch.mkdir()
            selected, pending = pending[:batch_size], pending[batch_size:]
            batch_paths, lines = zip(*selected)
            chunk = batch/'merizo.raw.tsv'
            chunk.write_text('\n'.join(lines)+'\n')
            inherited = batch/'chopping_merizo.txt'
            launch([sys.executable, CONSENSUS/'filter_domains.py', chunk, '-o', inherited],
                   env, batch, batch/'merizo_filter.log')
            active = cpu_pool.submit(method, 'unidoc', batch_paths, batch/'unidoc', env,
                                     workers=workers, inherited=inherited, unidoc_backend=unidoc_backend)
            submitted.append(batch)
            if first_submit is None:
                first_submit = time.perf_counter() - lane_start
        if now_done and active is None and not pending:
            break
        time.sleep(0.2 if now_done else 0.5)

    merged = output/'chopping_unidoc.txt'
    merged.write_text(''.join((batch/'chopping_unidoc.txt').read_text() for batch in submitted))
    check_rows(merged, set(by_id), 6)
    return {'merizo': merizo_seconds, 'unidoc': uni_seconds,
            'unidoc_batches': len(submitted), 'unidoc_first_submit_seconds': first_submit,
            'unidoc_finished_seconds': time.perf_counter()-lane_start}


def run_shard(paths, output, uuid, workers, profile, variant, overlap, gpu_slots,
              cpu_pool, batch_size, packed, unidoc_backend='fast'):
    output.mkdir()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
    elapsed = {}
    gpu_args = dict(profile=profile, variant=variant, gpu_slots=gpu_slots)

    def run_chainsaw(after=None):
        if after is not None:
            after.result()
        return method('chainsaw', paths, output/'chainsaw', env, **gpu_args)

    with ThreadPoolExecutor(max_workers=2) as pool:
        chainsaw = pool.submit(run_chainsaw) if overlap else None
        if batch_size:
            merizo = pool.submit(method, 'merizo', paths, output/'merizo', env, packed=packed, **gpu_args)
            if chainsaw is None:
                chainsaw = pool.submit(run_chainsaw, merizo)
            elapsed.update(stream_unidoc(merizo, paths, output, env, workers, cpu_pool, batch_size, unidoc_backend))
        else:
            elapsed['merizo'] = method('merizo', paths, output/'merizo', env, packed=packed, **gpu_args)
            uni = cpu_pool.submit(method, 'unidoc', paths, output/'unidoc', env, workers=workers,
                                  unidoc_backend=unidoc_backend)
            if chainsaw is None:
                chainsaw = pool.submit(run_chainsaw)
            elapsed['unidoc'] = uni.result()
        elapsed['chainsaw'] = chainsaw.result()
    start = time.perf_counter()
    launch([sys.executable, CONSENSUS/'get_consensus.py', '-c', 'chopping_merizo.txt',
            'chopping_chainsaw.txt', 'chopping_unidoc.txt', '-o', 'raw.tsv'],
           env, output, output/'consensus.log')
    launch([sys.executable, CONSENSUS/'filter_domains_consensus.py', 'raw.tsv', '-o', 'consensus.tsv'],
           env, output, output/'consensus_filter.log')
    check_rows(output/'consensus.tsv', {p.stem for p in paths}, 9)
    elapsed['consensus'] = time.perf_counter() - start
    return elapsed


def gpu_ids(indices):
    lines = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid',
                                     '--format=csv,noheader'], text=True)
    by_index = dict(line.split(', ') for line in lines.splitlines())
    return [by_index[str(i)] for i in indices]


def split_paths(paths, count):
    shards = [[] for _ in range(count)]
    load = [0] * count
    weighted = [(p.stat().st_size, p) for p in paths]
    for size, path in sorted(weighted, key=lambda item: (-item[0], item[1].name)):
        index = min(range(count), key=load.__getitem__)
        shards[index].append(path)
        load[index] += size
    return shards


def read_shards(manifest, paths, gpus):
    """Restore the exact input order and physical GPU assignment of an earlier run."""
    by_id = {p.stem: p for p in paths}
    assigned = {gpu: [] for gpu in gpus}
    seen = set()
    with manifest.open(newline='') as stream:
        reader = csv.DictReader(stream, delimiter='\t')
        if not {'afdb_model_id', 'gpu'} <= set(reader.fieldnames or []):
            raise ValueError('Shard manifest must contain afdb_model_id and gpu columns.')
        for row in reader:
            model = row['afdb_model_id']
            gpu = int(row['gpu'])
            if model not in by_id or model in seen or gpu not in assigned:
                raise ValueError(f'Invalid shard assignment for {model} on GPU {gpu}')
            assigned[gpu].append(by_id[model])
            seen.add(model)
    if seen != by_id.keys() or not all(assigned.values()):
        raise ValueError('Shard manifest must assign every input once and populate every GPU.')
    return list(assigned.values())


def merge_outputs(paths, folders, output):
    ids = {p.stem for p in paths}
    for filename in OUTPUTS:
        rows = sorted(line for folder in folders for line in (folder/filename).read_text().splitlines())
        merged = output/filename
        merged.write_text('\n'.join(rows)+'\n')
        check_rows(merged, ids, 9 if filename == 'consensus.tsv' else 6)


def run_gpu(paths, output, uuid, args):
    jobs = min(args.structures_per_gpu, len(paths))
    # Limit GPU subprocesses across all lanes on this physical card.
    slots = BoundedSemaphore(max(jobs, 2 if args.overlap else 1))
    cpu_jobs = min(jobs, args.unidoc_workers)
    cpu_base, extra = divmod(args.unidoc_workers, cpu_jobs)
    with ThreadPoolExecutor(max_workers=cpu_jobs) as cpu_pool:
        if jobs == 1:
            return run_shard(paths, output, uuid, args.unidoc_workers, args.profile,
                             args.variant, args.overlap, slots, cpu_pool, args.unidoc_batch_size,
                             args.packed_confidence, args.unidoc_backend)
        output.mkdir()
        folders = [output/f'worker{i:02d}' for i in range(jobs)]
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            futures = [
                pool.submit(run_shard, shard, folder, uuid, cpu_base + (i % cpu_jobs < extra),
                            args.profile, args.variant, args.overlap, slots, cpu_pool, args.unidoc_batch_size,
                            args.packed_confidence, args.unidoc_backend)
                for i, (shard, folder) in enumerate(zip(split_paths(paths, jobs), folders))
            ]
            timings = [future.result() for future in futures]
    merge_outputs(paths, folders, output)
    return {folder.name: timing for folder, timing in zip(folders, timings)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('-i', '--input', type=Path, required=True)
    p.add_argument('-o', '--output', type=Path, required=True)
    p.add_argument('--gpus', nargs='+', type=int, required=True)
    p.add_argument('--unidoc-backend', choices=('fast', 'reference'), default='fast',
                   help='UniDoc input preparation and CPU scheduling (default: fast).')
    p.add_argument('--unidoc-workers', type=int, default=4, help='CPU worker budget per GPU.')
    p.add_argument('--structures-per-gpu', type=int, default=1,
                   help='Number of independent structure workers per GPU (default: 1).')
    p.add_argument('--unidoc-batch-size', type=int, default=0,
                   help='Stream this many completed Merizo rows at a time to UniDoc (0 disables).')
    p.add_argument('--overlap', action='store_true', help='Allow Merizo and Chainsaw to overlap.')
    p.add_argument('--packed-confidence', action='store_true', help='Experimental packed Merizo confidence GRU.')
    p.add_argument('--profile', action='store_true')
    p.add_argument('--variant', choices=('v1', 'v2', 'v3', 'v4', 'ted-fast', 'tensorcore'), default='ted-fast')
    p.add_argument('--shard-manifest', type=Path, help='TSV containing afdb_model_id and physical gpu columns.')
    p.add_argument('--export-domains', action='store_true',
                   help='Export high and medium domains and verify their PDBs on disk.')
    args = p.parse_args()
    total_start = time.perf_counter()
    paths = sorted(args.input.resolve().glob('*.pdb'))
    if len(set(args.gpus)) != len(args.gpus) or len(paths) < len(args.gpus):
        p.error('Provide at least one PDB per GPU and use distinct GPU indices.')
    if args.unidoc_workers < 1 or args.structures_per_gpu < 1 or args.unidoc_batch_size < 0:
        p.error('Worker counts must be positive and the UniDoc batch size must be nonnegative.')
    ids = gpu_ids(args.gpus)
    output = args.output.resolve()
    shards = (read_shards(args.shard_manifest, paths, args.gpus) if args.shard_manifest
              else split_paths(paths, len(ids)))
    output.mkdir(parents=True)
    start = time.perf_counter()
    folders = [output/f'gpu{gpu}' for gpu in args.gpus]
    with ThreadPoolExecutor(max_workers=len(ids)) as pool:
        futures = [pool.submit(run_gpu, inputs, folder, uuid, args)
                   for inputs, folder, uuid in zip(shards, folders, ids)]
        stages = [future.result() for future in futures]
    merge_outputs(paths, folders, output)
    report = {'wall_seconds': time.perf_counter() - start, 'inputs': len(paths),
              'gpus': dict(zip(args.gpus, ids)), 'stages_seconds': dict(zip(args.gpus, stages)),
              'unidoc_workers_per_gpu': args.unidoc_workers, 'unidoc_backend': args.unidoc_backend,
              'structure_workers_per_gpu': args.structures_per_gpu,
              'unidoc_batch_size': args.unidoc_batch_size, 'packed_confidence': args.packed_confidence,
              'profile_enabled': args.profile, 'variant': args.variant,
              'overlap_gpu_models': args.overlap,
              'chainsaw_disable_tf32': os.environ.get('TED_CHAINSAW_DISABLE_TF32') == '1'}
    if args.export_domains:
        from export_domains import export_domains
        export_start = time.perf_counter()
        report['export'] = export_domains(paths, output/'consensus.tsv', output)
        report['export_seconds'] = time.perf_counter() - export_start
        report['end_to_end_seconds'] = time.perf_counter() - total_start
    (output/'timing.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
