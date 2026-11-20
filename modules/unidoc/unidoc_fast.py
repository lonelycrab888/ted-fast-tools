"""TED UniDoc input preparation and CPU scheduling for AFDB PDBs."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import re
import subprocess
from tempfile import TemporaryDirectory

from natsort import natsorted
from pdbtools import pdb_selres, pdb_tofasta

BINDIR = Path(__file__).resolve().parent / 'upstream/bin'
RANGE = re.compile(r'([0-9]{1,4})(?:-([0-9]{1,4}))?')


def selected_records(pdb, records, chopping):
    """Apply TED's closed residue ranges with pdb-tools' record semantics."""
    numbers = set()
    for part in chopping.replace(',', '_').split('_'):
        match = RANGE.fullmatch(part)
        if match:
            start = int(match[1])
            end = int(match[2] or match[1])
            if start <= end:
                numbers.update(range(start, end + 1))
                continue
        # Preserve pdb-tools' treatment of unusual range specifications.
        option = '-' + chopping.replace('-', ':').replace('_', ',')
        source, numbers = pdb_selres.check_input([option, str(pdb)])
        return pdb_selres.run(source, numbers)
    return pdb_selres.run(records, numbers)


def write_padded(destination, records):
    with destination.open('w', newline='') as out:
        for line in records:
            body = line.rstrip('\r\n')
            out.write(body.ljust(80) + '\n' if len(body) < 22 else line)


def prepare_pdb(pdb, chopping, destination):
    crop = chopping not in {'NULL', 'NO_SS'}
    # pdb_selres reads text with universal newlines. Uncropped PDBs retain
    # their original line endings except for the upstream short-record fix.
    with pdb.open(newline=None if crop else '') as source:
        records = source.readlines()
    fasta = pdb_tofasta.run(records, multi=None)
    next(fasta)  # FASTA header
    sequence = ''.join(fasta).replace('\n', '')
    selected = selected_records(pdb, records, chopping) if crop else records
    write_padded(destination, selected)
    if crop and destination.stat().st_size == 0:
        with pdb.open(newline='') as source:
            write_padded(destination, source)
    return hashlib.md5(sequence.encode('utf-8')).hexdigest(), len(sequence)


def predict(pdb, chopping, work, env):
    with TemporaryDirectory(prefix='structure-', dir=work) as tmp:
        directory = Path(tmp)
        prepared = directory/'input.pdb'
        md5, length = prepare_pdb(pdb, chopping, prepared)
        error = None
        try:
            with (directory/'stride.ss').open('wb') as ss:
                subprocess.run([BINDIR/'stride', prepared.name, '-rA'], env=env, cwd=directory,
                               stdout=ss, stderr=subprocess.PIPE, check=True)
            result = subprocess.run([BINDIR/'UniDoc_structure', prepared.name, 'A', 'stride.ss'],
                                    env=env, cwd=directory, capture_output=True, check=True)
            cuts = result.stdout.decode().replace('~', '-').replace(',', '_').replace('/', ',').rstrip('\n')
            domains = natsorted(cuts.split(',')) if cuts else []
            boundaries = ','.join(domains) or 'NULL'
            count = len(domains)
        except (OSError, subprocess.CalledProcessError, UnicodeError) as exc:
            detail = (getattr(exc, 'stderr', b'') or b'').decode(errors='replace').strip()
            error = f'UNIDOC_ERROR\t{pdb.stem}\t{exc}\t{detail}'
            boundaries, count = 'NO_SS', 0
    return f'{pdb.stem}\t{md5}\t{length}\t{count}\t{boundaries}\t1.00000\n', error


def run_batch(paths, inherited, raw, workers, env, log):
    for name in ('stride', 'UniDoc_structure'):
        if not (BINDIR / name).is_file():
            raise FileNotFoundError(f'Missing {BINDIR / name}. Restore the files with scripts/assets.py.')
    choppings = {}
    if inherited is not None:
        with inherited.open() as handle:
            for number, line in enumerate(handle, 1):
                row = line.rstrip().split('\t')
                if len(row) != 6 or row[0] in choppings:
                    raise ValueError(f'Malformed or duplicate chopping at {inherited}:{number}')
                choppings[row[0]] = row[4]
        missing = {p.stem for p in paths} - choppings.keys()
        if missing:
            raise ValueError(f'Missing inherited chopping for {sorted(missing)[:5]}')

    def run_one(pdb):
        return predict(pdb, choppings.get(pdb.stem, 'NULL'), raw.parent, env)

    # Schedule an individual PDB as soon as a CPU slot is free. map retains
    # input order in the TSV even when shorter PDBs finish first.
    with ThreadPoolExecutor(max_workers=min(workers, len(paths))) as pool, raw.open('w') as out:
        for row, error in pool.map(run_one, paths):
            out.write(row)
            if error is not None:
                print(error, file=log, flush=True)
