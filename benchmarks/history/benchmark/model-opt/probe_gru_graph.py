"""Measure capture, replay and eager costs for Merizo's IPA GRU shape."""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch


def seconds(fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return time.perf_counter() - start


@torch.no_grad()
def probe(length):
    gru = torch.nn.GRU(512, 256, num_layers=2, batch_first=True, bidirectional=True).cuda().eval()
    inputs = [torch.randn(1, length, 512, device='cuda') for _ in range(6)]
    static = inputs[0].clone()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            gru(static)
    torch.cuda.current_stream().wait_stream(side)
    ref = [gru(x)[0] for x in inputs]
    g = torch.cuda.CUDAGraph()
    def capture():
        nonlocal result
        with torch.cuda.graph(g, stream=side):
            result = gru(static)[0]
    result = None
    capture_s = seconds(capture)
    def replay_six():
        for x in inputs:
            static.copy_(x)
            g.replay()
    def eager_six():
        for x in inputs:
            gru(x)
    equal = []
    for x, expected in zip(inputs, ref):
        static.copy_(x)
        g.replay()
        equal.append(torch.equal(result, expected))
    eager, replay = [], []
    for _ in range(5):
        eager.append(seconds(eager_six))
        replay.append(seconds(replay_six))
    e, r = statistics.median(eager), statistics.median(replay)
    return dict(length=length, capture_seconds=capture_s, eager_six_seconds=eager, replay_six_seconds=replay,
                eager_median=e, replay_median=r, six_exact=all(equal),
                break_even_calls=6*capture_s/(e-r) if e > r else None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('length', type=int)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    torch.manual_seed(73)
    try:
        result = probe(args.length)
    except RuntimeError as exc:
        result = {'length': args.length, 'error': str(exc)}
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(result, flush=True)
