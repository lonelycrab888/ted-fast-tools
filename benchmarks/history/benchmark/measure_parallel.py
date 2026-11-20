"""Measure TED wall time and GPU-level utilization for structure worker counts."""
import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

BASE = Path(__file__).resolve().parents[1]


def measure(gpu, inputs, root, workers, overlap, batch_size, packed):
    label = f'w{workers}' + ('-overlap' if overlap else '')
    if batch_size:
        label += f'-stream{batch_size}'
    if packed:
        label += '-packed'
    destination = root / label
    command = [sys.executable, BASE/'run_fast.py', '-i', inputs, '-o', destination,
               '--gpus', str(gpu), '--structures-per-gpu', str(workers)]
    if overlap:
        command.append('--overlap')
    if batch_size:
        command.extend(('--unidoc-batch-size', str(batch_size)))
    if packed:
        command.append('--packed-confidence')
    sample_command = ['nvidia-smi', '-i', str(gpu), '--query-gpu=utilization.gpu,memory.used',
                      '--format=csv,noheader,nounits']
    with (root/f'{label}.log').open('w') as log, (root/f'{label}-gpu.csv').open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['seconds', 'gpu_util_pct', 'gpu_memory_mib'])
        start = time.monotonic()
        runner = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                  env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
        samples = []
        while runner.poll() is None:
            try:
                sample = subprocess.check_output(sample_command, text=True, timeout=5).strip()
                utilization, memory = map(int, sample.split(','))
                samples.append((utilization, memory))
                writer.writerow([round(time.monotonic()-start, 3), utilization, memory])
                file.flush()
            except (subprocess.SubprocessError, ValueError):
                pass
            time.sleep(1)
        report = {'workers': workers, 'overlap': overlap, 'unidoc_batch_size': batch_size, 'packed_confidence': packed, 'returncode': runner.returncode,
                  'sample_count': len(samples),
                  'mean_util_pct': statistics.mean(u for u, _ in samples) if samples else None,
                  'peak_memory_mib': max(m for _, m in samples) if samples else None}
        if runner.returncode == 0:
            timing = json.loads((destination/'timing.json').read_text())
            report['seconds'] = timing['wall_seconds']
            report['inputs'] = timing['inputs']
        with (root/'metrics.jsonl').open('a') as report_file:
            report_file.write(json.dumps(report)+'\n')
        print(label, json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('gpu', type=int)
    parser.add_argument('inputs', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--config', nargs='+', required=True,
                        help='Use N for sequential methods or No for overlapping methods.')
    parser.add_argument('--stream', type=int, default=0)
    parser.add_argument('--packed-confidence', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for config in args.config:
        measure(args.gpu, args.inputs, args.output, int(config.rstrip('o')), config.endswith('o'), args.stream, args.packed_confidence)


if __name__ == '__main__':
    main()
