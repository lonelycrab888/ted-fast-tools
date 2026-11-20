"""Compare the two fused IPA operations with the installed PyTorch operations."""
import argparse
import json
from pathlib import Path
import statistics
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from merizo_ipa_fast import point_logits, weighted_values


def measure(fn):
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(5):
        start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        start.record()
        result = fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
        del result
    return dict(median_ms=statistics.median(times), peak_extra_bytes=torch.cuda.max_memory_allocated()-before)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    rows = []
    for n in (1, 2, 3, 7, 31, 63, 127, 128, 200, 255, 256, 257, 513, 811, 1024, 1399, 1400):
        torch.manual_seed(n)
        q = torch.randn(1, n, 16, 4, 3, device='cuda')
        kv = torch.randn(1, n, 16, 12, 3, device='cuda')
        k, v = kv.split((4, 8), dim=-2)
        w = torch.rand(16, device='cuda')
        a = torch.randn(1, 16, n, n, device='cuda').softmax(-1)
        variants = {
            'distance': (
                lambda: (sum(((q.unsqueeze(-4)-k.unsqueeze(-5))**2).unbind(-1))*w[:, None]).sum(-1)*-0.5,
                lambda: point_logits(q, k, w)),
            'values': (
                lambda: (a[..., None, :, :, None]*v.permute(0, 2, 4, 1, 3)[..., None, :, :]).sum(-2),
                lambda: weighted_values(a, v))}
        row = {'length': n}
        for name, (old, new) in variants.items():
            ref, actual = old(), new()
            info = {'equal': torch.equal(ref, actual), 'different': (ref != actual).sum().item(),
                    'max_abs': (ref-actual).abs().max().item()}
            del ref, actual
            if n in (128, 513, 1400):
                info['reference'], info['fast'] = measure(old), measure(new)
            row[name] = info
        rows.append(row)
        print(n, row['distance']['different'], row['values']['different'], flush=True)
    args.output.write_text(json.dumps(rows, indent=2)+'\n')
    if not all(r[k]['equal'] for r in rows for k in ('distance','values')):
        raise SystemExit('mismatching tensors in the compiled kernel')
if __name__ == '__main__':
    main()
