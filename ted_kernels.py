"""Faster local kernels for the same three-method TED workflow."""
from ctypes import CDLL, POINTER, c_double, c_int
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def install_chainsaw():
    from src.domain_assignment import assigners, sparse_lowrank

    original = sparse_lowrank.greedy_V
    library = CDLL(str(Path(__file__).resolve().parent / 'modules/chainsaw/chainsaw_assign.so'))
    assign = library.assign_mse
    assign.argtypes = [POINTER(c_double), c_int, c_int, c_int, POINTER(c_int)]
    assign.restype = None

    def greedy_mse(Y, N_iters=3, K_init=4, cost_type='mse'):
        if cost_type != 'mse':
            return original(Y, N_iters, K_init, cost_type)
        symmetric = (Y + Y.T) / 2
        pair = np.ascontiguousarray(symmetric, dtype=np.float64)
        labels = np.empty(pair.shape[0], dtype=np.int32)
        assign(pair.ctypes.data_as(POINTER(c_double)), len(labels),
               N_iters, K_init, labels.ctypes.data_as(POINTER(c_int)))
        domains = np.unique(labels[labels >= 0])
        assignment = (labels[:, None] == domains[None, :]).astype(int)
        same_domain = (labels[:, None] == labels[None, :]) & (labels[:, None] >= 0)
        np.subtract(same_domain, pair, out=pair)
        np.square(pair, out=pair)
        return assignment, pair.sum()

    sparse_lowrank.greedy_V = greedy_mse
    assigners.greedy_V = greedy_mse


def separate_components(domain_map, z, domain_ids, distance=8.0):
    import torch

    nres = domain_ids.numel()
    adjacency = ((domain_map != 0) & (z.reshape(nres, nres) <= distance)).cpu().numpy()
    graph = csr_matrix(adjacency)
    _, labels = connected_components(graph, directed=False)
    # NetworkX leaves a vertex with a self-loop in the graph. Truly isolated
    # vertices are removed, and components are numbered by their first vertex.
    active = np.diff(graph.indptr) != 0
    _, first, inverse = np.unique(labels[active], return_index=True, return_inverse=True)
    order = np.argsort(np.argsort(first)) + 1
    result = np.zeros(nres, dtype=np.int64)
    result[active] = order[inverse]
    return torch.as_tensor(result, dtype=domain_ids.dtype, device=domain_ids.device)


def install_merizo():
    from model.utils import features, utils
    from model.ipa import nndef_ipa_primitives
    from scipy.spatial.distance import cdist

    features.distance_matrix = cdist
    # The upstream inference loader immediately replaces every parameter via
    # strict checkpoint loading. Avoid generating unused random initial values.
    nndef_ipa_primitives.trunc_normal_init_ = lambda weights, **kwargs: None
    utils.separate_components = separate_components
    utils.clean_domains = clean_domains
    upstream_singletons = utils.clean_singletons

    def compatible_singletons(labels, threshold):
        if len(labels) <= threshold:
            return upstream_singletons(labels, threshold)
        try:
            return clean_singletons(labels, threshold)
        except IndexError:
            return upstream_singletons(labels, threshold)

    utils.clean_singletons = compatible_singletons
    utils.remap_ids = remap_ids


def clean_domains(dom_ids, min_num):
    labels = dom_ids.cpu().numpy().copy()
    counts = np.bincount(labels)
    small = (labels != 0) & (counts[labels] < min_num)
    small[0] = False
    source = np.maximum.accumulate(np.where(~small, np.arange(len(labels)), 0))
    dom_ids.copy_(dom_ids.new_tensor(labels[source]))
    return dom_ids


def clean_singletons(dom_ids, threshold):
    labels = dom_ids.cpu().numpy().copy()
    stops = np.flatnonzero(np.r_[labels[1:] != labels[:-1], True]) + 1
    sizes = np.diff(np.r_[0, stops])
    lengths = np.ones_like(labels)
    # Reproduce the original counter and slice behavior, including isolated
    # single residues whose stored length can differ at run boundaries.
    start = last = 0
    counter = 1
    for end, size in zip(stops, sizes):
        if size > 1:
            last = end - 1
            counter += size - 1
        lengths[start:last + 1] = counter
        start = end
        counter = 0
    if np.any(lengths[:threshold] >= threshold):
        labels[:threshold] = labels[threshold]
    if np.any(lengths[-threshold:] >= threshold):
        labels[-threshold:] = labels[-threshold-1]
    short = lengths < threshold
    long = np.flatnonzero(~short)
    if len(long):
        short[:long[0]] = False
        short[long[-1]:] = False
    transitions = np.flatnonzero(np.diff(np.r_[False, short, False]))
    for start, stop in zip(transitions[::2], transitions[1::2]):
        mid = (start + stop - 1) // 2
        labels[start:mid+1] = labels[start-1]
        labels[mid:stop] = labels[stop]
    dom_ids.copy_(dom_ids.new_tensor(labels))
    return dom_ids


def remap_ids(dom_ids):
    labels = dom_ids.cpu().numpy()
    active = labels != 0
    _, first, inverse = np.unique(labels[active], return_index=True, return_inverse=True)
    mapped = np.zeros_like(labels)
    mapped[active] = (np.argsort(np.argsort(first)) + 1)[inverse]
    return dom_ids.new_tensor(mapped)


def install_afdb_input():
    from model.utils import features

    old_parse = features.pdb_to_features

    def parse(file, resi=None, zipped=False):
        # AFDB PDBs ordinarily contain exactly N, CA, C, O for every residue.
        # Keep the general parser for incomplete, alternate or selected inputs.
        if resi is not None or zipped:
            return old_parse(file, resi=resi, zipped=zipped)
        models, remarks = features.open_pdb(file)
        records = models[0]
        if not len(records) or len(records) % 4:
            return old_parse(file)
        groups = records.reshape(-1, 4)
        indices = groups['resi']
        if (not np.all(groups['n'] == ['N', 'CA', 'C', 'O']) or
                np.any(groups['alt'] != '') or
                not np.all(indices == np.arange(1, len(groups)+1)[:, None]) or
                not np.all(groups['chain'] == groups['chain'][0, 0])):
            return old_parse(file)
        # The pretrained code sorts by residue ID and uses the remaining
        # structured fields to resolve ties. Keep the identical array order.
        ordered = np.sort(records, order='resi')
        return ordered, ordered[ordered['n'] == 'CA'], ordered, remarks

    features.pdb_to_features = parse


def install_chainsaw_io():
    from src import featurisers
    from src.utils.pdb_reres import renumber_residues

    def renumber(source, output):
        with open(source) as before, open(output, 'w') as after:
            after.writelines(renumber_residues(before, 1))

    featurisers.renum_pdb_file = renumber
