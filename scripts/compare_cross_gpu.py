"""Compare TED stage tables and exported PDBs across GPU architectures."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

METHODS = ("merizo", "chainsaw", "unidoc")


def read_rows(path, width):
    rows = {}
    with path.open(newline="") as stream:
        for line, row in enumerate(csv.reader(stream, delimiter="\t"), 1):
            if len(row) != width or not row[0] or row[0] in rows:
                raise ValueError(f"{path}:{line}: malformed or duplicate model row")
            rows[row[0]] = row[1:]
    return rows


def compare_table(reference, candidate, consensus=False):
    keys = sorted(reference.keys() & candidate.keys())
    differences = {k: {"4090": reference[k], "5090": candidate[k]}
                   for k in keys if reference[k] != candidate[k]}
    result = {
        "reference": len(reference), "candidate": len(candidate),
        "missing": sorted(reference.keys() - candidate.keys()),
        "extra": sorted(candidate.keys() - reference.keys()),
        "identical": len(keys) - len(differences),
        "different_inputs": [k for k in keys if reference[k][:2] != candidate[k][:2]],
        "differences": differences,
    }
    if consensus:
        result["same_high_medium"] = sum(
            reference[k][2:4] == candidate[k][2:4] and reference[k][5:7] == candidate[k][5:7]
            for k in keys
        )
    else:
        result["same_boundaries"] = sum(reference[k][2:4] == candidate[k][2:4] for k in keys)
        result["same_score"] = sum(reference[k][4] == candidate[k][4] for k in keys)
        result["max_score_delta"] = max(
            (abs(float(reference[k][4]) - float(candidate[k][4])) for k in keys), default=0.0
        )
    return result


def compare_pdbs(reference, candidate):
    expected = {}
    for line in (reference / "domains.sha256").read_text().splitlines():
        digest, filename = line.split(maxsplit=1)
        if filename in expected:
            raise ValueError(f"Duplicate reference domain: {filename}")
        expected[filename] = digest
    domain_dir = candidate / "domains"
    if not domain_dir.is_dir():
        raise FileNotFoundError(f"{domain_dir} is missing. Run TED with --export-domains.")
    actual = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in domain_dir.glob("*.pdb")}
    return {
        "reference": len(expected), "candidate": len(actual),
        "identical": sum(actual.get(k) == v for k, v in expected.items()),
        "different": sorted(k for k in expected.keys() & actual.keys() if expected[k] != actual[k]),
        "missing": sorted(expected.keys() - actual.keys()), "extra": sorted(actual.keys() - expected.keys()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="Archived TSVs and domains.sha256")
    parser.add_argument("candidate", type=Path, help="TED results including domains/")
    parser.add_argument("report", type=Path, help="Output JSON including all differing rows")
    args = parser.parse_args()
    reference = read_rows(args.reference / "consensus.tsv", 9)
    if not reference:
        parser.error("The reference consensus table contains no structures.")
    report = {}
    for method in METHODS:
        name = f"chopping_{method}.txt"
        rows = read_rows(args.reference / name, 6)
        if rows.keys() != reference.keys():
            raise ValueError(f"Reference {name} and consensus.tsv contain different model IDs")
        report[name] = compare_table(rows, read_rows(args.candidate / name, 6))
    consensus = compare_table(reference, read_rows(args.candidate / "consensus.tsv", 9), consensus=True)
    report["consensus.tsv"] = consensus
    pdbs = compare_pdbs(args.reference, args.candidate)
    report["domain_pdb"] = pdbs
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    for name, entry in report.items():
        print(f"{name}: {entry['identical']}/{entry['reference']} exact")
    print(f"High/medium boundaries: {consensus['same_high_medium']}/{len(reference)}")
    if (any(r["missing"] or r["extra"] or r.get("different_inputs") for r in report.values())
            or consensus["same_high_medium"] != len(reference) or pdbs["different"]):
        raise SystemExit(f"Retained-domain verification failed. See {args.report}")


if __name__ == "__main__":
    main()
