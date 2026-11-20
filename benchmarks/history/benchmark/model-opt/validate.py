"""Compare upstream method rows, filtered outputs and the final consensus by model ID."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare import compare, METHODS


def raw_rows(root, method):
    if method == 'unidoc':
        paths = sorted(root.glob('gpu*/unidoc-batches/batch*/unidoc/raw.tsv'))
    else:
        paths = []
    if not paths:
        paths = sorted(root.glob(f'gpu*/{method}/raw.tsv'))
    if not paths:
        paths = [root/method/'raw.tsv']
    results = {}
    for path in paths:
        for line in path.read_text().splitlines():
            name = line.split('\t', 1)[0]
            if name == 'chain_id':
                continue
            if name in results:
                raise ValueError(f'Duplicate raw ID {name} in {path}')
            results[name] = line
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('reference', type=Path, help='Reference filtered files and consensus')
    p.add_argument('reference_raw', type=Path, help='Reference run directory with method stages')
    p.add_argument('candidate', type=Path)
    p.add_argument('output', type=Path)
    args = p.parse_args()
    filtered = compare(args.reference, args.candidate)
    raw = {}
    for method in METHODS:
        old, new = raw_rows(args.reference_raw, method), raw_rows(args.candidate, method)
        ids = old.keys() | new.keys()
        different = sorted(k for k in ids if old.get(k) != new.get(k))
        raw[method] = {'rows': len(ids), 'identical': len(ids)-len(different), 'different': different}
    rows = [line.split('\t') for line in (args.candidate/'consensus.tsv').read_text().splitlines()]
    report = {'structures': len(rows), 'raw': raw, 'filtered_and_consensus': filtered,
              'high': sum(int(r[3]) for r in rows), 'medium': sum(int(r[4]) for r in rows),
              'without_high_medium': sorted(r[0] for r in rows if int(r[3])+int(r[4]) == 0)}
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(args.candidate, len(rows), 'raw matches', [s['identical'] for s in raw.values()],
          'filtered matches', [s['rows_identical'] for s in filtered.values()], flush=True)
    if (not all(s['rows_identical']==len(rows) for s in filtered.values())
            or not all(s['identical']==s['rows']==len(rows) for s in raw.values())):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
