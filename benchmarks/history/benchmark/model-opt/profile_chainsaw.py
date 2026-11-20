"""Separate queued CNN GPU work, CPU domain assignment and worker STRIDE time."""
import argparse
from collections import defaultdict
from functools import wraps
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, '/home/zhr/tools/ted-tools/ted_consensus_1.0/programs/chainsaw')
from src import featurisers
from src.domain_assignment.assigners import SparseLowRank
from src.models.rosetta import trRosettaNetwork
import run_method


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--channels-last', action='store_true')
    args = p.parse_args()
    args.output.mkdir(parents=True)
    intervals, host = [], defaultdict(list)
    cnn = trRosettaNetwork.forward
    def forward(net, x):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        if args.channels_last:
            if not getattr(net, '_layout_ready', False):
                net.to(memory_format=torch.channels_last)
                net._layout_ready = True
            x = x.contiguous(memory_format=torch.channels_last)
        result = cnn(net, x)
        end.record()
        intervals.append((start,end))
        return result
    trRosettaNetwork.forward = forward

    def timed(name, fn):
        @wraps(fn)
        def measure(*a, **kw):
            start = time.perf_counter()
            try:
                return fn(*a, **kw)
            finally:
                host[name].append(time.perf_counter()-start)
        return measure
    SparseLowRank.assign_domains = timed('cpu_domain_assignment',SparseLowRank.assign_domains)
    featurisers.calculate_ss = timed('stride_workers',featurisers.calculate_ss)
    with TemporaryDirectory(dir=args.output) as directory:
        for file in args.input.glob('*.pdb'):
            shutil.copy2(file, Path(directory)/file.name)
        sys.argv = ['run_method.py', 'chainsaw', '--variant', 'v4', '--structure_directory', directory,
                    '-o', str(args.output/'raw.tsv'), '--append']
        start = time.perf_counter()
        run_method.main()
        torch.cuda.synchronize()
        wall = time.perf_counter()-start
    report = {'channels_last':args.channels_last,'wall_seconds':wall,'cnn_cuda':{'calls':len(intervals),
        'stream_seconds':sum(a.elapsed_time(b) for a,b in intervals)/1000},
        'host_seconds':{name: {'calls':len(values),'sum':sum(values)} for name,values in host.items()}}
    (args.output/'profile.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report)


if __name__ == '__main__':
    main()
