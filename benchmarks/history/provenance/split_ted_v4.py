"""Split the frozen HumanMotifbench gallery using the published TED v4 boundaries."""
import argparse
from collections import Counter, defaultdict
import csv
from hashlib import sha256
import json
from pathlib import Path
import re
from itertools import chain
import sys

ARCHIVE_NAME = 'ted_365m_domain_boundaries_consensus_level.tsv.gz'
ARCHIVE_MD5 = '6fb6837a40a41e0fbb10191c77ec5d71'
ARCHIVE_BYTES = 2885163510
ARCHIVE_URL = f'https://zenodo.org/records/13908086/files/{ARCHIVE_NAME}'
MODEL = re.compile(r'AF-[A-Z0-9]+(?:-[0-9]+)?-F([1-9][0-9]*)-model_v4')
DOMAIN = re.compile(r'(AF-[A-Z0-9]+(?:-[0-9]+)?-F[1-9][0-9]*-model_v4)_TED[0-9]+')
SEGMENT = re.compile(r'([1-9][0-9]*)-([1-9][0-9]*)')


def read_inputs(bench):
    dataset = bench / 'datasets/plddt70_v1'
    gallery = dataset / 'gallery.json'
    manifest = json.loads((dataset / 'manifest.json').read_text())
    gallery_bytes = gallery.read_bytes()
    gallery_sha = sha256(gallery_bytes).hexdigest()
    if manifest['version'] != 'plddt70_v1' or gallery_sha != manifest['files_sha256']['gallery.json']:
        raise ValueError('Frozen gallery version or SHA-256 differs from manifest')
    rows = json.loads(gallery_bytes)
    by_model = {}
    for row in rows:
        model = row['structure_id'].removeprefix('hu_')
        match = MODEL.fullmatch(model)
        if match is None or model in by_model:
            raise ValueError(f'Unexpected or duplicate model ID: {model}')
        by_model[model] = (row, int(match[1]))
    if len(by_model) != manifest['structures'] or len({r['accession'] for r in rows}) != manifest['accessions']:
        raise ValueError('Frozen gallery count differs from manifest')
    return by_model, gallery_sha


def read_boundaries(path, gallery):
    by_model = defaultdict(list)
    seen = set()
    with path.open(newline='') as handle:
        for domain_id, boundaries, consensus in csv.reader(handle, delimiter='\t'):
            match = DOMAIN.fullmatch(domain_id)
            if match is None or match[1] not in gallery or domain_id in seen:
                raise ValueError(f'Unexpected or duplicate TED domain: {domain_id}')
            if consensus not in {'high', 'medium'}:
                raise ValueError(f'Unexpected consensus for {domain_id}: {consensus}')
            segments = []
            for part in boundaries.split('_'):
                segment = SEGMENT.fullmatch(part)
                if segment is None:
                    raise ValueError(f'Malformed boundary for {domain_id}: {part}')
                start, end = map(int, segment.groups())
                if start > end or (segments and start <= segments[-1][1]):
                    raise ValueError(f'Invalid segment ordering for {domain_id}: {boundaries}')
                segments.append((start, end))
            seen.add(domain_id)
            by_model[match[1]].append((domain_id, boundaries, consensus, segments))
    if not seen:
        raise ValueError('No matched domains found in the official table')
    return by_model


def read_parent_pdb(path):
    atoms = defaultdict(list)
    ca = {}
    chains = set()
    for line in path.read_bytes().splitlines(keepends=True):
        if not line.startswith(b'ATOM  '):
            continue
        if len(line) < 66 or line[26:27] != b' ' or line[16:17] != b' ':
            raise ValueError(f'Unexpected short record, insertion code or altloc in {path.name}')
        chains.add(line[21:22])
        position = int(line[22:26])
        atoms[position].append(line)
        if line[12:16].strip() == b'CA':
            if position in ca:
                raise ValueError(f'Duplicate CA position {position} in {path.name}')
            ca[position] = float(line[60:66])
    if len(chains) != 1 or list(atoms) != list(range(1, len(atoms) + 1)) or atoms.keys() != ca.keys():
        raise ValueError(f'{path.name} does not have a single chain numbered consecutively from 1 with one CA per residue')
    return atoms, ca


def write_domain(path, positions, atoms):
    records = tuple(chain.from_iterable(atoms[n] for n in positions))
    atom_payload = b''.join(records)
    payload = atom_payload + b'TER\nEND\n'
    with path.open('xb') as handle:
        handle.write(payload)
    # Verify that the stored ATOM records are byte-identical to the selected parent records.
    saved = path.read_bytes()
    saved_atoms = saved.splitlines(keepends=True)[:-2]
    if tuple(saved_atoms) != records:
        raise ValueError(f'ATOM record round-trip mismatch in {path.name}')
    if {int(line[22:26]) for line in saved_atoms if line[12:16].strip() == b'CA'} != set(positions):
        raise ValueError(f'Residue boundary mismatch in {path.name}')
    return len(records), sha256(saved).hexdigest()


def split(bench, stage, archive):
    if archive.name != ARCHIVE_NAME or archive.stat().st_size != ARCHIVE_BYTES:
        raise ValueError('Archive filename or size differs from official Zenodo release')
    gallery, gallery_sha = read_inputs(bench)
    source_gallery = bench.parents[1] / 'HUMAN/manifests/human_gallery.json'
    source_bytes = source_gallery.read_bytes()
    original = {r['structure_id']: r for r in json.loads(source_bytes)}
    selected = read_boundaries(stage / 'source_boundaries.tsv', gallery)
    pdb_root = stage / 'domains'
    pdb_root.mkdir()
    per_fragment = defaultdict(Counter)
    domain_stats = Counter()
    matched_accessions = set()
    gallery_accessions = set()
    unmatched_accessions = set()
    errors = []
    with (stage / 'domains.tsv').open('w', newline='') as domain_file, \
         (stage / 'structures.tsv').open('w', newline='') as structure_file, \
         (stage / 'unmatched.tsv').open('w', newline='') as unmatched_file:
        domains = csv.writer(domain_file, delimiter='\t', lineterminator='\n')
        structures = csv.writer(structure_file, delimiter='\t', lineterminator='\n')
        unmatched = csv.writer(unmatched_file, delimiter='\t', lineterminator='\n')
        domains.writerow(['domain_id', 'structure_id', 'accession', 'afdb_model_id', 'boundaries', 'consensus', 'residues', 'atoms', 'mean_ca_plddt', 'pdb_path', 'pdb_sha256'])
        structures.writerow(['structure_id', 'accession', 'afdb_model_id', 'fragment', 'status', 'domains', 'parent_residues', 'assigned_residues', 'reason'])
        unmatched.writerow(['structure_id', 'accession', 'afdb_model_id', 'fragment', 'reason'])
        for index, (model, (row, fragment)) in enumerate(gallery.items(), 1):
            sid = row['structure_id']
            accession = row['accession']
            gallery_accessions.add(accession)
            counts = per_fragment['F1' if fragment == 1 else 'F2+']
            counts['total'] += 1
            domain_rows = selected.get(model, [])
            if not domain_rows:
                reason = 'no_published_high_or_medium_domain_for_exact_model_id'
                structures.writerow([sid, accession, model, fragment, 'unmatched', 0, '', '', reason])
                unmatched.writerow([sid, accession, model, fragment, reason])
                counts['unmatched'] += 1
                unmatched_accessions.add(accession)
                continue
            try:
                parent = original[sid]
                pdb = Path(parent['path'])
                if parent['accession'] != accession or pdb.stem != model:
                    raise ValueError(f'Parent manifest mismatch: {pdb}')
                atoms, ca = read_parent_pdb(pdb)
                prepared = []
                for domain_id, boundaries, consensus, segments in sorted(domain_rows):
                    if segments[-1][1] > len(atoms):
                        raise ValueError(f'{domain_id} ends at {segments[-1][1]} beyond parent length {len(atoms)}')
                    positions = list(chain.from_iterable(range(a, b + 1) for a, b in segments))
                    prepared.append((domain_id, boundaries, consensus, segments, positions))
            except (OSError, ValueError, KeyError) as error:
                counts['invalid'] += 1
                errors.append({'structure_id': sid, 'error': str(error)})
                structures.writerow([sid, accession, model, fragment, 'invalid', 0, '', '', str(error)])
                continue

            assigned = set()
            for domain_id, boundaries, consensus, segments, positions in prepared:
                rel = Path('domains') / f'{domain_id}.pdb'
                count, pdb_sha = write_domain(stage / rel, positions, atoms)
                mean = sum(ca[i] for i in positions) / len(positions)
                domains.writerow([domain_id, sid, accession, model, boundaries, consensus, len(positions), count, f'{mean:.4f}', rel.as_posix(), pdb_sha])
                assigned.update(positions)
                domain_stats['total'] += 1
                domain_stats[consensus] += 1
                domain_stats['discontinuous'] += len(segments) > 1
                domain_stats['atoms'] += count
                domain_stats['domain_residues_including_overlaps'] += len(positions)
            matched_accessions.add(accession)
            counts['matched'] += 1
            counts['parents_fully_assigned'] += len(assigned) == len(atoms)
            counts['parent_residues'] += len(atoms)
            counts['assigned_residues'] += len(assigned)
            structures.writerow([sid, accession, model, fragment, 'matched', len(prepared), len(atoms), len(assigned), ''])
            if counts['matched'] % 2000 == 0:
                print(f'Processed {index:,}/{len(gallery):,} candidates, wrote {domain_stats["total"]:,} domains', flush=True)

    pdb_count = sum(p.is_file() for p in pdb_root.iterdir())
    source_domain_count = sum(map(len, selected.values()))
    passed = not errors and pdb_count == domain_stats['total'] == source_domain_count and sum(c['matched'] for c in per_fragment.values()) == len(selected)
    summary = {
        'dataset': 'plddt70_v1_ted_v4',
        'parent_gallery_sha256': gallery_sha,
        'human_structure_manifest_sha256': sha256(source_bytes).hexdigest(),
        'ted_release': {'zenodo_record': 13908086, 'url': ARCHIVE_URL, 'md5': ARCHIVE_MD5, 'bytes': ARCHIVE_BYTES},
        'policy': 'Exact AFDB v4 model-ID match, published high and medium consensus only, unmodified selected ATOM records with original fragment-local numbering.',
        'candidate_structures': len(gallery),
        'candidate_accessions': len(gallery_accessions),
        'matched_structures': sum(c['matched'] for c in per_fragment.values()),
        'matched_accessions': len(matched_accessions),
        'unmatched_structures': sum(c['unmatched'] for c in per_fragment.values()),
        'accessions_with_unmatched_structures': len(unmatched_accessions),
        'accessions_with_no_matched_structures': len(gallery_accessions - matched_accessions),
        'invalid_structures': len(errors),
        'by_fragment': {f: dict(c) for f,c in per_fragment.items()},
        'domains': dict(domain_stats),
    }
    validation = {
        'passed': passed,
        'published_boundary_rows_matched': source_domain_count,
        'pdb_files': pdb_count,
        'pdb_atom_records_and_residue_sets_verified': domain_stats['total'],
        'input_and_output_sha256': {
            name: sha256((stage / name).read_bytes()).hexdigest()
            for name in ['source_boundaries.tsv', 'domains.tsv', 'structures.tsv', 'unmatched.tsv', 'split_ted_v4.py']
        },
        'errors': errors,
    }
    for name, report in [('summary.json', summary), ('validation.json', validation)]:
        (stage / name).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    print(f'Validation passed: {passed}', flush=True)
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bench', type=Path)
    parser.add_argument('--print-model-ids', action='store_true')
    parser.add_argument('--stage', type=Path)
    parser.add_argument('--archive', type=Path)
    args = parser.parse_args()
    if args.print_model_ids:
        gallery, _ = read_inputs(args.bench)
        print('\n'.join(gallery))
        return 0
    if args.stage is None or args.archive is None:
        parser.error('--stage and --archive are required to split domains')
    return split(args.bench, args.stage, args.archive)


if __name__ == '__main__':
    sys.exit(main())
