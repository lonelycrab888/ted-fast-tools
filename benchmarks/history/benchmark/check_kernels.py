"""Differential checks for the local Merizo and Chainsaw kernels."""
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

BASE = Path(__file__).resolve().parents[1]
TED = Path('/home/zhr/tools/ted-tools/ted_consensus_1.0')
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(TED/'programs/merizo'))
sys.path.insert(0, str(TED/'programs/chainsaw'))
import ted_kernels as fast
from model.utils import utils
from src.domain_assignment import sparse_lowrank

rng = np.random.default_rng(44)
report = {}
for name, args in [('clean_domains', [25]), ('clean_singletons', [5]), ('remap_ids', [])]:
    total = edge_exceptions = 0
    for _ in range(600):
        n = rng.integers(6, 300)
        labels = np.repeat(rng.integers(0, 12, n), rng.integers(1, 13, n))[:n]
        ts = torch.tensor(labels)
        try:
            before = getattr(utils, name)(ts.clone(), *args)
        except IndexError:
            edge_exceptions += 1
            continue
        after = getattr(fast, name)(ts.clone(), *args)
        assert before.equal(after), (name, labels, before, after)
        total += 1
    report[name] = {'identical': total, 'upstream_index_errors': edge_exceptions}

for n in (10, 80, 300):
    for _ in range(20):
        labels = torch.tensor(rng.integers(0, 9, n))
        domain_map = utils.instance_matrix(labels.float())[0]
        positions = torch.tensor(rng.normal(size=(n, 3))*10)
        dm = torch.cdist(positions, positions).reshape(1, n, n, 1)
        assert utils.separate_components(domain_map, dm, labels).equal(
            fast.separate_components(domain_map, dm, labels))
report['connected_components'] = {'identical': 60}

original = sparse_lowrank.greedy_V
fast.install_chainsaw()
report['chainsaw_assignment'] = []
for n in (128, 400, 800, 1400):
    slots = rng.integers(0, 10, size=n)
    matrix = (rng.random((n, n))*0.35 + (slots[:, None] == slots[None, :])*0.6).astype(np.float32)
    start = time.perf_counter()
    a, loss = original(matrix)
    old = time.perf_counter()-start
    start = time.perf_counter()
    b, new_loss = sparse_lowrank.greedy_V(matrix)
    new = time.perf_counter()-start
    assert np.array_equal(a, b) and np.isclose(loss, new_loss, rtol=1e-7)
    report['chainsaw_assignment'].append(dict(length=n, reference_seconds=old, fast_seconds=new,
                                             speedup=old/new, assignment_identical=True))
payload = json.dumps(report, indent=2)+'\n'
(BASE/'benchmark/kernel_checks.json').write_text(payload)
print(payload)
