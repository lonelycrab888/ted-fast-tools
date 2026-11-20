"""Exercise all three module CLIs and the integrated pipeline against archived predictions."""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/"examples/pdb"
EXPECTED = ROOT/"examples/expected"


def run(script, *args, cwd, cpu=False, blackwell=False):
    env = os.environ.copy()
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    if blackwell:
        env.pop("NVIDIA_TF32_OVERRIDE", None)
        env["TED_CHAINSAW_DISABLE_TF32"] = "1"
    if cpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
    command = [ROOT/script] if script.endswith(".sh") else [sys.executable, ROOT/script]
    subprocess.run([*command, *map(str, args)], check=True, cwd=cwd, env=env)


def equal_table(actual, expected, *, ignore_score=False):
    def rows(path):
        lines = path.read_text().splitlines()
        return sorted(line.rsplit("\t", 1)[0] for line in lines) if ignore_score else sorted(lines)
    if rows(actual) != rows(expected):
        raise ValueError(f"Mismatch between {actual} and {expected}")


def check(base, gpus, platform):
    blackwell = platform in ("blackwell", "5090")
    for name in ("merizo", "chainsaw", "unidoc"):
        out = base/name
        options = ["--gpu", gpus[0]] if name != "unidoc" else [
            "--inherit",base/"merizo/chopping_merizo.txt","--workers",2]
        run(f"modules/{name}/run.py","-i",DATA,"-o",out,*options,cwd=base, cpu=name=="unidoc", blackwell=blackwell)
        equal_table(out/f"chopping_{name}.txt", EXPECTED/f"chopping_{name}.txt",
                    ignore_score=blackwell and name != "unidoc")
    single = DATA/"AF-P0CG08-F1-model_v4.pdb"
    run("modules/unidoc/run.py","-i",single,"-o",base/"whole","--whole-chain",cwd=base,cpu=True)
    equal_table(base/"whole/chopping_unidoc.txt", EXPECTED/"whole_chain_P0CG08.tsv")
    if (base/"whole/unidoc/run.log").read_text().strip():
        raise ValueError("Inspect whole/unidoc/run.log for a UniDoc error")
    out = base/"full"
    entry = "run-blackwell.sh" if blackwell else "run_fast.py"
    run(entry,"-i",DATA,"-o",out,"--gpus",*gpus,"--overlap","--unidoc-batch-size",32,
        "--variant","ted-fast","--export-domains",cwd=base)
    if blackwell:
        run("scripts/compare_cross_gpu.py", EXPECTED, out, base/"cross-gpu.json", cwd=base)
        for name in ("merizo", "chainsaw", "unidoc"):
            filename = f"chopping_{name}.txt"
            equal_table(base/name/filename, out/filename)
        print("PASS: independent methods, whole-chain UniDoc and retained domains on RTX 5090.")
        return
    for ref in [*EXPECTED.glob("chopping_*.txt"), EXPECTED/"consensus.tsv"]:
        equal_table(out/ref.name,ref)
    actual = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (out/"domains").glob("*.pdb")}
    expected = dict((name, sha) for sha, name in (line.split() for line in (EXPECTED/"domains.sha256").read_text().splitlines()))
    if actual != expected:
        raise ValueError("Exported domain set or PDB content differs from reference")
    print(f"PASS: independent methods, whole-chain UniDoc, consensus and {len(actual)} domain PDB hashes.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpus", type=int, nargs="+", required=True)
    p.add_argument("--platform", choices=("4090", "blackwell", "5090"), default="4090",
                   help="4090 checks every reference field; blackwell reports score/low differences and checks retained domains.")
    p.add_argument("--output", type=Path, help="Optional new directory for keeping test outputs.")
    args = p.parse_args()
    if args.output:
        args.output.mkdir(parents=True, exist_ok=False)
        check(args.output.resolve(), args.gpus, args.platform)
    else:
        with TemporaryDirectory(prefix="ted-smoke-") as tmp:
            check(Path(tmp), args.gpus, args.platform)
