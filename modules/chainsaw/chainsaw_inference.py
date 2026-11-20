"""Exact CPU shortcuts for the pinned Chainsaw inference path."""
from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist


@dataclass
class ChainGeometry:
    source: str
    sequence: str
    coords: np.ndarray


def read_chain_a(path, three_to_one):
    """Use the directly observable CA geometry for an ordered, standard chain A."""
    if not str(path).endswith('.pdb'):
        return None
    previous = None
    coordinates = []
    sequence = []
    has_ca = False
    with open(path, 'rb') as handle:
        for line in handle:
            if line.startswith((b'ENDMDL', b'END   ')):
                break
            if line.startswith(b'HETATM'):
                return None
            if not line.startswith(b'ATOM  '):
                continue
            if len(line) < 54 or line[16:17] != b' ' or line[21:22] != b'A' or line[26:27] != b' ':
                return None
            residue = line[17:27]
            if residue != previous:
                if previous is not None and not has_ca:
                    return None
                try:
                    if int(line[22:26]) != len(sequence) + 1:
                        return None
                    sequence.append(three_to_one[line[17:20].decode('ascii')])
                except (ValueError, KeyError):
                    return None
                previous = residue
                has_ca = False
            if line[12:16].strip() == b'CA':
                if has_ca:
                    return None
                try:
                    coordinates.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                except ValueError:
                    return None
                has_ca = True
    if not coordinates or not has_ca:
        return None
    # Bio.PDB casts its parsed coordinates to float32 before scipy computes distances.
    return ChainGeometry(path, ''.join(sequence), np.asarray(coordinates, dtype=np.float32))


def component_count(matrix, residues):
    selected = matrix[np.ix_(residues, residues)]
    selected = selected[np.any(selected, axis=1)]
    if len(selected) == 0:
        return 0
    if np.any((selected != 0) & (selected != 1)):
        # Preserve the original row key for non-binary caller-supplied features.
        return len({''.join(str(int(x)) for x in row) for row in selected})
    return len(np.unique(np.packbits(selected.astype(bool), axis=1), axis=0))


def install_chainsaw_inference():
    from Bio.PDB.Polypeptide import protein_letters_3to1
    from src import domain_chop, featurisers
    from ted_kernels import install_chainsaw_io

    install_chainsaw_io()
    parse = featurisers.get_model_structure
    get_sequence = featurisers.get_model_structure_sequence
    get_distance = featurisers.get_distance
    pair_labels = domain_chop.make_pair_labels
    remove_few = domain_chop.PairwiseDomainPredictor.remove_domains_with_few_ss_components

    def geometry(path):
        fast = read_chain_a(path, protein_letters_3to1)
        return parse(path) if fast is None else fast

    def sequence(model, chain='A'):
        if isinstance(model, ChainGeometry):
            if chain == 'A':
                return model.sequence
            model = parse(model.source)
        return get_sequence(model, chain=chain)

    def distances(model, chain='A'):
        if isinstance(model, ChainGeometry):
            if chain == 'A':
                pair = cdist(model.coords, model.coords)
                pair[pair == 0] = pair[pair > 0].min()
                return (pair ** -1)[None]
            model = parse(model.source)
        return get_distance(model, chain=chain)

    def recycle_labels(n_res, domain_dict, id_string=None, save_dir=None, non_aligned_residues=()):
        if save_dir is not None or len(non_aligned_residues):
            return pair_labels(n_res, domain_dict, id_string, save_dir, non_aligned_residues)
        labels = np.zeros((n_res, n_res))
        for name, residues in domain_dict.items():
            if name != 'linker' and len(residues):
                index = np.asarray(residues, dtype=np.intp)
                labels[np.ix_(index, index)] = 1
        return labels

    def filter_components(model, domains, features):
        if getattr(model, 'ss_mod', False):
            return remove_few(model, domains, features)
        kept = {}
        for name, residues in domains.items():
            if name == 'linker' or not len(residues):
                continue
            residues = sorted(residues)
            count = component_count(features[1], residues) + component_count(features[2], residues)
            if count < model.min_ss_components:
                domains['linker'] += residues
            else:
                kept[name] = residues
        kept['linker'] = domains['linker']
        return kept

    featurisers.get_model_structure = geometry
    featurisers.get_model_structure_sequence = sequence
    featurisers.get_distance = distances
    domain_chop.make_pair_labels = recycle_labels
    domain_chop.PairwiseDomainPredictor.remove_domains_with_few_ss_components = filter_components
