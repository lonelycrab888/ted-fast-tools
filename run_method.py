"""Run one pinned TED method using the isolated optimizations in this directory."""
import argparse
import os
from contextlib import contextmanager, nullcontext
import runpy
from pathlib import Path
import sys
import time

MODULES = Path(__file__).resolve().parent / 'modules'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('method', choices=('chainsaw', 'merizo'))
    parser.add_argument('--profile', type=Path)
    parser.add_argument('--cuda-profile', type=Path)
    parser.add_argument('--packed-confidence', action='store_true')
    parser.add_argument('--variant', choices=('v1', 'v2', 'v3', 'v4', 'ted-fast', 'tensorcore'), default='ted-fast')
    args, upstream_args = parser.parse_known_args()
    # Scope the driver override to this one predictor, before torch initializes CUDA.
    if args.method == 'chainsaw' and os.environ.get('TED_CHAINSAW_DISABLE_TF32') == '1':
        os.environ['NVIDIA_TF32_OVERRIDE'] = '0'
    if args.method != 'merizo' and (args.cuda_profile or args.packed_confidence):
        parser.error('GRU options apply to Merizo only.')
    program, script = {
        'merizo': ('merizo', 'predict_afdb.py'),
        'chainsaw': ('chainsaw', 'get_predictions.py'),
    }[args.method]
    path = MODULES / program / 'upstream' / script
    sys.path.insert(0, str(path.parent))
    sys.argv = [str(path), *upstream_args]
    if args.method == 'merizo':
        from model.utils import utils
        from ted_kernels import install_merizo, separate_components
        if args.variant == "v1":
            utils.separate_components = separate_components
        else:
            install_merizo()
            if args.variant in ('v3', 'v4', 'ted-fast'):
                from ted_kernels import install_afdb_input
                install_afdb_input()
            if args.variant in ('v4', 'ted-fast'):
                from modules.merizo.merizo_ipa_fast import install_merizo_ipa
                install_merizo_ipa()
            if args.variant == 'ted-fast':
                from modules.merizo.merizo_gru_graphs import install_gru_graphs
                install_gru_graphs()
    else:
        from ted_kernels import install_chainsaw
        install_chainsaw()
        if args.variant != 'v1':
            from scipy.spatial.distance import cdist
            from src import featurisers
            featurisers.distance_matrix = cdist
        if args.variant == 'v3':
            from ted_kernels import install_chainsaw_io
            install_chainsaw_io()
        if args.variant in ('v4', 'ted-fast'):
            from modules.chainsaw.chainsaw_inference import install_chainsaw_inference
            install_chainsaw_inference()
    if args.variant == 'tensorcore':
        import torch
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    features = nullcontext()
    if args.method == 'chainsaw' and args.variant in ('v4', 'ted-fast'):
        from modules.chainsaw.chainsaw_stride import prefetch_stride
        features = prefetch_stride(upstream_args)
    with features, cuda_profile(args.cuda_profile, args.packed_confidence):
        execute(path, args.profile)


@contextmanager
def cuda_profile(output, packed):
    if packed:
        from modules.merizo.merizo_gru import install_packed_confidence
        install_packed_confidence()
    if output is None:
        yield
        return
    from modules.merizo.merizo_gru import install_cuda_profile
    save = install_cuda_profile()
    start = time.perf_counter()
    yield
    save(output, time.perf_counter()-start)


def execute(path, profile_output):
    if profile_output:
        import cProfile
        profile = cProfile.Profile()
        try:
            profile.runcall(runpy.run_path, str(path), run_name='__main__')
        finally:
            profile.dump_stats(profile_output)
    else:
        runpy.run_path(str(path), run_name='__main__')


if __name__ == '__main__':
    main()
