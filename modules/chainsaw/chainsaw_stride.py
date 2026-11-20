"""Bounded STRIDE look-ahead for a fresh Chainsaw directory inference."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch


@contextmanager
def prefetch_stride(argv):
    from src import constants, featurisers

    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--structure_directory', type=Path)
    parser.add_argument('-o', '--output', type=Path)
    parser.add_argument('--ss_mod', action='store_true')
    args, _ = parser.parse_known_args(argv)
    directory = args.structure_directory
    if args.ss_mod or directory is None or args.output is None or args.output.exists() or not directory.is_dir():
        yield
        return

    directory = directory.resolve()
    paths = iter(directory / name for name in os.listdir(directory)
                 if name.endswith('.pdb') and (directory / name).is_file())
    make_features = featurisers.inference_time_create_features
    with TemporaryDirectory(dir=args.output.resolve().parent, prefix='chainsaw-stride-') as scratch:
        with ThreadPoolExecutor(max_workers=2) as pool:
            pending = {}

            def prepare(path):
                # Use the same renumberer, chain and STRIDE executable as the pinned method.
                renumbered = Path(scratch) / (path.stem + '_renum.pdb')
                ss = Path(scratch) / (path.name + '_ss.txt')
                try:
                    featurisers.renum_pdb_file(str(path), str(renumbered))
                    featurisers.calculate_ss(str(renumbered), 'A', constants.STRIDE_EXE, ssfile=str(ss))
                finally:
                    renumbered.unlink(missing_ok=True)
                return ss

            def fill():
                while len(pending) < 4:
                    path = next(paths, None)
                    if path is None:
                        break
                    pending[path] = pool.submit(prepare, path)

            def features(pdb_path, chain='A', secondary_structure=True, renumber_pdbs=True,
                         add_recycling=True, add_mask=False, stride_path=constants.STRIDE_EXE,
                         ss_mod=False, *, model_structure=None):
                source = Path(pdb_path)
                path = source.parent.resolve() / source.name
                if (path not in pending or chain != 'A' or not secondary_structure
                        or not renumber_pdbs or stride_path != constants.STRIDE_EXE or ss_mod
                        or model_structure is None):
                    return make_features(pdb_path, chain, secondary_structure, renumber_pdbs,
                                         add_recycling, add_mask, stride_path, ss_mod,
                                         model_structure=model_structure)
                job = pending.pop(path)
                fill()
                distance = featurisers.get_distance(model_structure)
                length = distance.shape[-1]
                ss = job.result()
                try:
                    helix, strand = featurisers.make_ss_matrix(str(ss), nres=length)
                finally:
                    ss.unlink(missing_ok=True)
                channels = 3 + 2 * int(add_recycling) + int(add_mask)
                stacked = np.zeros((1, channels, length, length), dtype=np.float32)
                stacked[0, 0] = distance[0]
                stacked[0, 1] = helix
                stacked[0, 2] = strand
                return torch.from_numpy(stacked)

            fill()
            featurisers.inference_time_create_features = features
            try:
                yield
            finally:
                featurisers.inference_time_create_features = make_features
