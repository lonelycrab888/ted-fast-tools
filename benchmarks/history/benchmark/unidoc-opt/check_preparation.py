"""Compare the PDB delivered to the official STRIDE binary and its sequence metadata."""
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
import os
from pathlib import Path
import runpy
import shlex
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parents[1]
sys.path.insert(0, str(BASE))
from unidoc_fast import BINDIR, prepare_pdb, predict

WRAPPER = BINDIR.parent/'Run_UniDoc_from_scratch_structure_afdb.py'


def reference(pdb, domain, folder, capture):
    """Capture STRIDE input for byte comparison, or call both real binaries."""
    copy = folder/pdb.name
    inherited = folder/'merizo.tsv'
    out = folder/'reference.tsv'
    shutil.copyfile(pdb, copy)
    inherited.write_text(f'{pdb.stem}\tmd5\t1\t1\t{domain}\t1.00000\n')
    command = subprocess.check_output
    expected_hash = None

    def call(cmd, *args, **kwargs):
        nonlocal expected_hash
        exe, *options = shlex.split(cmd)
        if capture and exe == str(BINDIR/'stride'):
            expected_hash = sha256(Path(options[0]).read_bytes()).hexdigest()
            return b''
        if capture and exe == str(BINDIR/'UniDoc_structure'):
            return b'1~1\n'
        return command(cmd, *args, **kwargs)

    argv = [str(WRAPPER), '-i', str(copy), '--inherit_chopping', str(inherited), '--out', str(out)]
    with patch.object(sys, 'argv', argv), patch.object(subprocess, 'check_output', call):
        runpy.run_path(str(WRAPPER), run_name='__main__')
    return out.read_text(), expected_hash


def check(item):
    path, chopping = item
    with TemporaryDirectory(prefix='ted-unidoc-check-') as tmp:
        folder = Path(tmp)
        before, expected_hash = reference(path, chopping, folder, capture=True)
        prepared = folder/'fast.pdb'
        md5, size = prepare_pdb(path, chopping, prepared)
        assert before.split('\t')[1:3] == [md5, str(size)], path
        assert expected_hash == sha256(prepared.read_bytes()).hexdigest(), (path, chopping)
        return path.name


def main():
    bench = BASE/'benchmark'
    inherited = (bench/'holdout-reference/chopping_merizo.txt').read_text().splitlines()
    choppings = {row[0]: row[-2] for line in inherited if (row := line.split('\t'))}
    paths = sorted((bench/'holdout').glob('*.pdb'), key=lambda p: -p.stat().st_size)
    with ProcessPoolExecutor(max_workers=4) as pool:
        checked = list(pool.map(check, ((p, choppings[p.stem]) for p in paths)))

    with TemporaryDirectory(prefix='ted-unidoc-fixtures-') as tmp:
        folder = Path(tmp)
        atom = next(row for row in paths[0].read_text().splitlines() if row.startswith('ATOM'))
        def record(kind, number, chain='A', insertion=' ', name='ALA'):
            return f'{kind:<6}'+atom[6:17]+name+' '+chain+f'{number:4d}'+insertion+atom[27:]+'\r\n'
        custom = folder/'mixed.pdb'
        custom.write_bytes(('HEADER\r\n'+record('ATOM', 1)+record('ATOM', 2, insertion='A')+
                            record('ANISOU', 2)+record('HETATM', 3, name='MSE')+
                            record('ATOM', 1, chain='B')+record('TER', 3)+'END\r\n').encode())
        cases = [(custom, s) for s in ('NULL', 'NO_SS', '1-2', '1-1_3-3', '2', '98-99')]
        bare = folder/'bare.pdb'
        bare.write_bytes(record('ATOM', 1).encode())
        cases.append((bare, '98-99'))  # empty selection falls back to the original PDB
        for item in cases:
            check(item)

        # A header-only selection should trigger the deployed STRIDE failure path.
        error_case = folder/'expected-error'
        error_case.mkdir()
        ref_row, _ = reference(custom, '98-99', error_case, capture=False)
        row, error = predict(custom, '98-99', error_case, os.environ.copy())
        assert row == ref_row and row.split('\t')[4] == 'NO_SS' and error is not None
        assert not list(error_case.glob('structure-*'))

    report = {'holdout_pdbs_byte_identical': len(checked),
              'sequence_md5_and_length_identical': len(checked),
              'edge_pdbs_byte_identical': len(cases), 'real_NO_SS_row_identical': True}
    assert len(checked) == 128
    (ROOT/'preparation-check.json').write_text(json.dumps(report, indent=2)+'\n')
    print(report)

if __name__ == '__main__':
    main()
