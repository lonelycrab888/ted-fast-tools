"""Bounded CUDA graphs for repeated, exactly matching Merizo IPA GRU shapes."""
import atexit
from collections import Counter
import json

import torch

from modules.merizo.runtime import supports_fast_path


def install_gru_graphs():
    from model.ipa.nndef_ipa import StructureModuleTransition

    initial = StructureModuleTransition.__init__

    def initialize(module, *args, **kwargs):
        initial(module, *args, **kwargs)
        for gru in module.layers:
            wrap_gru(gru)

    StructureModuleTransition.__init__ = initialize


def wrap_gru(gru):
    previous = gru.forward
    counts, replays = Counter(), Counter()
    entries = {}
    blocked = set()

    def forward(x, hx=None):
        if (hx is not None or gru.training or torch.is_grad_enabled() or torch.is_autocast_enabled()
                or not x.is_cuda or not supports_fast_path(x.device)
                or x.dtype != torch.float32 or x.ndim != 3 or x.shape[0] != 1
                or x.shape[1] < 512 or not x.is_contiguous()):
            return previous(x, hx)
        length = x.shape[1]
        stream = torch.cuda.current_stream(x.device)
        key = (x.device.index, stream.cuda_stream, length, gru.weight_ih_l0.data_ptr())
        counts[key] += 1
        if key not in entries:
            # One IPA forward calls this GRU six times. Only capture shapes
            # observed in another forward, with a fixed memory budget per GRU.
            if counts[key] <= 6 or key in blocked or len(entries) >= 2:
                return previous(x, hx)
            try:
                entries[key] = capture(previous, x, stream)
            except torch.cuda.OutOfMemoryError:
                blocked.add(key)
            if key not in entries:
                return previous(x, hx)
        graph, static, output, state = entries[key]
        static.copy_(x)
        graph.replay()
        replays[key] += 1
        # Do not expose graph-owned storage that a subsequent replay overwrites.
        return output.clone(), state.clone()

    def report():
        summary = {'eligible_calls': sum(counts.values()), 'replays': sum(replays.values()),
                   'captures': len(entries), 'replays_by_length': {str(k[2]): v for k,v in replays.items()}}
        print('Merizo GRU graphs ' + json.dumps(summary), flush=True)

    gru.forward = forward
    atexit.register(report)


def capture(forward, x, stream):
    with torch.cuda.device(x.device):
        static = torch.empty_like(x)
        static.copy_(x)
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(stream)
        with torch.cuda.stream(capture_stream):
            forward(static)
        stream.wait_stream(capture_stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            output, state = forward(static)
    return graph, static, output, state
