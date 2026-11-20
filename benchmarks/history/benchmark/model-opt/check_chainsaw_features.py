"""Cross-check C-alpha input, the complete STRIDE feature tensor and unusual PDBs."""
import argparse
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import FunctionType

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, '/home/zhr/tools/ted-tools/ted_consensus_1.0/programs/chainsaw')
from Bio.PDB.Polypeptide import protein_letters_3to1
from src import featurisers
from scipy.spatial.distance import cdist
from chainsaw_inference import ChainGeometry, read_chain_a, install_chainsaw_inference
from chainsaw_stride import prefetch_stride


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    featurisers.distance_matrix = cdist
    original_parse = featurisers.get_model_structure
    original_distance = featurisers.get_distance
    original_seq = featurisers.get_model_structure_sequence
    fn = featurisers.inference_time_create_features
    original_features = FunctionType(fn.__code__, dict(fn.__globals__), fn.__name__, fn.__defaults__, fn.__closure__)
    original_features.__kwdefaults__ = fn.__kwdefaults__
    install_chainsaw_inference()
    equal = []
    fast_chains = 0
    with TemporaryDirectory(dir=args.output.resolve().parent) as work:
        root = Path(work)
        for path in args.input.glob('*.pdb'):
            (root/path.name).symlink_to(path.resolve())
        # The inference scope normally uses private PDB copies. Reference sidecars
        # in this test are created beside the symlink in the temporary directory.
        names = [n for n in os.listdir(root) if n.endswith('.pdb')]
        with prefetch_stride(['--structure_directory', str(root), '-o', str(root/'output.tsv')]):
            for name in names:
                source = str(root/name)
                legacy = original_parse(source)
                fast = featurisers.get_model_structure(source)
                fast_chains += isinstance(fast, ChainGeometry)
                seq_ok = featurisers.get_model_structure_sequence(fast) == original_seq(legacy)
                dist_ok = (featurisers.get_distance(fast) == original_distance(legacy)).all()
                old = original_features(source, model_structure=legacy)
                new = featurisers.inference_time_create_features(source, model_structure=fast)
                equal.append({'model': Path(name).stem, 'seq': seq_ok, 'distance': bool(dist_ok),
                              'features': torch.equal(old, new)})
        # Use explicitly noncanonical PDBs to check the dispatch to the full parser.
        original = next(args.input.glob('*.pdb')).read_text().splitlines(keepends=True)
        atoms = [line for line in original if line.startswith('ATOM')][:15]
        ca = next(line for line in atoms if line[12:16].strip() == 'CA')
        def change(line, index, value):
            return line[:index]+value+line[index+len(value):]
        cases = {'altloc': [change(x,16,'A') for x in atoms],
                 'insertion': [change(x,26,'A') for x in atoms],
                 'other_chain': [change(x,21,'B') for x in atoms],
                 'hetatm': ['HETATM'+ca[6:]], 'missing_ca': [x for x in atoms if x[12:16].strip()!='CA'],
                 'duplicate_ca': [ca, ca], 'gap': [change(x,22,'   9') for x in atoms]}
        fallbacks = {}
        for case, lines in cases.items():
            path = root / (case+'.pdb')
            path.write_text(''.join(lines)+'END\n')
            fallbacks[case] = read_chain_a(str(path), protein_letters_3to1) is None
    report={'inputs': len(equal), 'fast_chains': fast_chains,'exact': sum(all((r['seq'],r['distance'],r['features'])) for r in equal),
            'fallbacks': fallbacks,'different': [r for r in equal if not all((r['seq'],r['distance'],r['features']))]}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(report)
    if report['exact'] != len(equal) or not all(fallbacks.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
