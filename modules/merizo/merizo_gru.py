"""Opt-in packed confidence GRU and CUDA-event profiling for upstream Merizo."""
from collections import defaultdict
import inspect
import json
import textwrap
import time

import torch
from torch.nn.utils.rnn import pack_sequence


def predict_confidence(decoder, domain_masks, dom_ids, unique_ids):
    confidence = torch.zeros_like(dom_ids).float()
    if len(unique_ids) == 0:
        return confidence
    if len(unique_ids) == 1:
        mask = dom_ids == unique_ids[0]
        state = decoder.conf_gru(domain_masks[:, mask])[1]
        confidence[mask] = decoder.conf_out(state[-1:]).flatten().clamp(0, 1)
        return confidence
    membership = dom_ids[None, :] == unique_ids[:, None]
    # Packing preserves the actual (unpadded) end of each sequence in both
    # directions. h_n is restored to input order by PyTorch.
    domains = [domain_masks[0, mask] for mask in membership]
    packed = pack_sequence(domains, enforce_sorted=False)
    state = decoder.conf_gru(packed)[1]
    scores = decoder.conf_out(state[-1]).flatten().clamp(0, 1)
    for mask, score in zip(membership, scores):
        confidence[mask] = score
    return confidence


def install_packed_confidence():
    from model.decoders import mask_decoder

    # Limit the replacement to the confidence block and fail if TED changes it.
    source = textwrap.dedent(inspect.getsource(mask_decoder.MaskTransformer.forward))
    first = source.index('    # Iterate over each pred index')
    last = source.index('    return dom_ids.reshape(-1), conf_res.reshape(-1)', first)
    if source[first:last].count('self.conf_gru(dom)') != 1:
        raise RuntimeError('Unsupported upstream Merizo confidence block')
    source = (source[:first]
              + '    conf_res = predict_confidence(self, domain_masks, dom_ids, unique_ids)\n'
              + source[last:])
    namespace = dict(vars(mask_decoder), predict_confidence=predict_confidence)
    exec(compile(source, inspect.getfile(mask_decoder.MaskTransformer), 'exec'), namespace)
    mask_decoder.MaskTransformer.forward = namespace['forward']


def install_cuda_profile():
    from model import network

    original = network.Merizo.__init__
    samples = defaultdict(list)
    calls = defaultdict(list)

    def attach(module, name):
        active = []

        def begin(_module, inputs, kwargs):
            x = inputs[0] if inputs else kwargs['s']
            if isinstance(x, tuple):
                x = x[0]
            if isinstance(x, torch.nn.utils.rnn.PackedSequence):
                x = x.data
            if not x.is_cuda:
                active.append(None)
                return
            torch.cuda.nvtx.range_push(name)
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            active.append((event, time.perf_counter()))

        def end(_module, _inputs, _result):
            stamp = active.pop()
            if stamp is not None:
                stop = torch.cuda.Event(enable_timing=True)
                stop.record()
                torch.cuda.nvtx.range_pop()
                samples[name].append((stamp[0], stop))
                calls[name].append(time.perf_counter()-stamp[1])

        module.register_forward_pre_hook(begin, with_kwargs=True)
        module.register_forward_hook(end)

    def initialize(model, *args, **kwargs):
        original(model, *args, **kwargs)
        attach(model, 'whole_network')
        attach(model.ipa.ipa, 'ipa_attention')
        attach(model.decoder_head, 'mask_decoder')
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.GRU):
                attach(module, name)

    network.Merizo.__init__ = initialize

    def save(path, wall_seconds):
        if not samples:
            raise RuntimeError('No GPU forward calls recorded.')
        torch.cuda.synchronize()
        report = {'profile_wall_seconds': wall_seconds, 'stages': {}}
        for name, intervals in samples.items():
            timings = [start.elapsed_time(end) for start, end in intervals]
            report['stages'][name] = {
                'calls': len(timings), 'stream_ms_total': sum(timings),
                'mean_stream_ms': sum(timings)/len(timings),
                'host_call_ms_total': sum(calls[name])*1000}
        path.write_text(json.dumps(report, indent=2)+'\n')

    return save
