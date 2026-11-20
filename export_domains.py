"""Write high/medium domain PDBs with original residue numbering and verify them on disk."""
import argparse
from collections import Counter, defaultdict
import csv
from hashlib import sha256
from itertools import chain
from pathlib import Path
from tempfile import TemporaryDirectory

# Parent parsing and byte-preserving PDB writing are shared with the validated exporter.
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


def export_domains(paths, consensus, output):
    parents = {p.stem: p for p in paths}
    counts = Counter(high=0, medium=0, with_domain=0, no_domain=0)
    seen = set()
    outputs = ("domains", "domains.tsv", "no_domain.tsv")
    if any((output/name).exists() for name in outputs):
        raise FileExistsError(f"Domain export already exists in {output}")
    with TemporaryDirectory(dir=output, prefix=".ted-export-") as temp:
        stage = Path(temp)
        (stage/"domains").mkdir()
        with (stage/"domains.tsv").open("w", newline="") as table, \
             (stage/"no_domain.tsv").open("w") as no_domain, consensus.open() as rows:
            manifest = csv.writer(table, delimiter="\t", lineterminator="\n")
            manifest.writerow(["domain_id","model_id","consensus","boundaries","residues","atoms",
                               "mean_ca_plddt","pdb_sha256","pdb_path"])
            no_domain.write("model_id\n")
            for row in csv.reader(rows, delimiter="\t"):
                model, _, length, nh, nm, _, high, medium, _ = row
                if model in seen or model not in parents:
                    raise ValueError(f"Unexpected or duplicate model {model}")
                seen.add(model)
                retained = []
                for level, number, cuts in (("high",nh,high), ("medium",nm,medium)):
                    domains = [] if cuts == "na" else cuts.split(",")
                    if len(domains) != int(number):
                        raise ValueError(f"Inconsistent {level} count for {model}")
                    retained.extend((level, cut) for cut in domains)
                if not retained:
                    counts["no_domain"] += 1
                    no_domain.write(model+"\n")
                    continue
                atoms, ca = read_parent_pdb(parents[model])
                if len(ca) != int(length):
                    raise ValueError(f"Parent length mismatch for {model}")
                counts["with_domain"] += 1
                for i, (level, cut) in enumerate(retained, 1):
                    positions = []
                    for segment in cut.split("_"):
                        lo, hi = map(int, segment.split("-"))
                        if not 1 <= lo <= hi <= len(ca) or hi-lo+1 < 5:
                            raise ValueError(f"Invalid segment {segment} in {model}")
                        positions.extend(range(lo, hi+1))
                    if len(positions) < 25 or positions != sorted(set(positions)):
                        raise ValueError(f"Invalid retained domain {cut} in {model}")
                    domain = f"{model}_TES{i:02d}"
                    rel = Path("domains")/f"{domain}.pdb"
                    natoms, digest = write_domain(stage/rel, positions, atoms)
                    mean = sum(ca[p] for p in positions)/len(positions)
                    manifest.writerow([domain,model,level,cut,len(positions),natoms,f"{mean:.4f}",digest,rel])
                    counts[level] += 1
        if seen != parents.keys():
            raise ValueError(f"Consensus is missing {len(parents.keys()-seen)} input structures")
        for name in outputs:
            (stage/name).rename(output/name)
    return dict(counts)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-i","--input", type=Path, required=True, help="Directory of original PDB files.")
    p.add_argument("--consensus", type=Path, required=True)
    p.add_argument("-o","--output", type=Path, required=True, help="New output directory.")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    print(export_domains(sorted(a.input.glob("*.pdb")), a.consensus, a.output))
