"""Compare TED stage files and final boundaries by model ID."""
import argparse
import json
from pathlib import Path

METHODS = ('merizo', 'chainsaw', 'unidoc')


def read(path):
    rows = [line.split('\t') for line in path.read_text().splitlines()]
    results = {r[0]: r[1:] for r in rows}
    if len(results) != len(rows):
        raise ValueError(f'Duplicate rows in {path}')
    return results


def compare(reference, candidate):
    report = {}
    for name in (*(f'chopping_{m}.txt' for m in METHODS), 'consensus.tsv'):
        before, after = read(reference / name), read(candidate / name)
        if before.keys() != after.keys():
            raise ValueError(f'Missing or extra model IDs for {candidate / name}')
        exact = [m for m in before if before[m] != after[m]]
        def geometry(row):
            return (row[2:4], row[5:7]) if name == 'consensus.tsv' else row[2:4]
        altered = [m for m in before if geometry(before[m]) != geometry(after[m])]
        report[name] = {'rows': len(before), 'rows_identical': len(before)-len(exact),
                        'geometry_identical': len(before)-len(altered),
                        'different_rows': exact, 'different_geometry': altered}
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('reference', type=Path)
    p.add_argument('candidates', type=Path, nargs='+')
    p.add_argument('-o', '--output', type=Path)
    args = p.parse_args()
    results = {str(d): compare(args.reference, d) for d in args.candidates}
    if args.output:
        args.output.write_text(json.dumps(results, indent=2)+'\n')
    for target, report in results.items():
        print(target)
        for name, rows in report.items():
            print('  ', name, rows['rows_identical'], rows['geometry_identical'], '/', rows['rows'])


if __name__ == '__main__':
    main()
