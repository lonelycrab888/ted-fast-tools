"""Check whether the other UniDoc executable in the installed archive is interchangeable."""
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from natsort import natsorted

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parents[1]
sys.path.insert(0, str(BASE))
from unidoc_fast import BINDIR, prepare_pdb


def run():
    ref = (BASE/'benchmark/holdout-reference/chopping_merizo.txt').read_text().splitlines()
    targets = sorted((int(r[2]), r[0], r[4]) for line in ref if (r := line.split('\t'))
                     if 200 <= int(r[2]) <= 750 and r[4] not in ('NULL', 'NO_SS'))
    picked = [targets[i*(len(targets)-1)//7] for i in range(8)]
    report = []
    for length, name, chopping in picked:
        with TemporaryDirectory(prefix='ted-unidoc-native-') as tmp:
            d = Path(tmp)
            (d/'stride').symlink_to(BINDIR/'stride')
            prepare_pdb(BASE/'benchmark/holdout'/f'{name}.pdb', chopping, d/'input.pdb')
            row = {'model': name, 'parent_length': length}
            with (d/'stride.ss').open('wb') as ss:
                subprocess.run([BINDIR/'stride', 'input.pdb', '-rA'], cwd=d, stdout=ss, check=True)
            for binary, args in [('UniDoc_structure', ['input.pdb','A','stride.ss']),
                                 ('UniDoc_struct', ['input.pdb','A'])]:
                try:
                    proc = subprocess.run([BINDIR/binary, *args], cwd=d, capture_output=True, timeout=20, check=True)
                    cuts = proc.stdout.decode().strip().replace('~','-').replace(',','_').split('/')
                    row[binary] = ','.join(natsorted(cuts))
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                    row[binary] = str(exc)
            report.append(row)
    result = {'inputs': len(report),
              'same_boundaries': sum(r['UniDoc_structure'] == r['UniDoc_struct'] for r in report),
              'results': report}
    (ROOT/'bundled-native-comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Bundled native comparison:',result['same_boundaries'],'/',len(report))

if __name__ == '__main__':
    run()
