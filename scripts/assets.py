"""Verify or restore the pinned model weights and native programs for this checkout."""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
CHAINSAW_STRIDE = "modules/chainsaw/upstream/stride/stride"
CHAINSAW_STRIDE_URL = (
    "https://raw.githubusercontent.com/psipred/ted-tools/"
    "6041723c28f1782c2cbb1a4edb9b3523c0a1739c/ted_consensus_1.0/programs/chainsaw/stride/stride"
)
UNIDOC_URL = "https://yanglab.qd.sdu.edu.cn/UniDoc/download/UniDoc_2023.tgz"
UNIDOC_SHA256 = "4cf2c4d5ada32a6218cc655bea1751e364cd6d7ecd747d8f26e8e0d73cc30805"
UNIDOC_MEMBERS = {
    "modules/unidoc/upstream/bin/stride": "UniDoc/bin/stride",
    "modules/unidoc/upstream/bin/UniDoc_structure": "UniDoc/bin/UniDoc_structure",
}


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def restore(stream, target, expected, executable):
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as out:
        temporary = Path(out.name)
    try:
        with temporary.open("wb") as out:
            shutil.copyfileobj(stream, out)
        if checksum(temporary) != expected:
            raise ValueError(f"SHA-256 mismatch for {target}")
        temporary.chmod(0o755 if executable else 0o644)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def download(url):
    print(f"Downloading {url}", flush=True)
    try:
        return urlopen(url, timeout=90)
    except URLError as error:
        raise SystemExit(f"Could not download {url}: {error}") from error


def open_unidoc_archive(resources):
    directory = resources.enter_context(tempfile.TemporaryDirectory())
    path = Path(directory) / "UniDoc_2023.tgz"
    with download(UNIDOC_URL) as src, path.open("wb") as dest:
        shutil.copyfileobj(src, dest)
    if checksum(path) != UNIDOC_SHA256:
        raise ValueError(f"SHA-256 mismatch for {UNIDOC_URL}")
    return resources.enter_context(tarfile.open(path, "r:gz"))


def member_stream(archive, name):
    member = archive.getmember(name)
    if not member.isfile():
        raise ValueError(f"Expected regular file: {name}")
    return archive.extractfile(member)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    op = p.add_mutually_exclusive_group()
    op.add_argument("--bundle", type=Path, help="Restore files from the offline tar archive.")
    op.add_argument("--download", action="store_true",
                    help="Download and verify weights and Linux x86_64 native programs.")
    op.add_argument("--download-native", action="store_true",
                    help="Download and verify only the Linux x86_64 native programs.")
    op.add_argument("--download-weights", action="store_true",
                    help="Download and verify only the .pt weights from pinned upstream URLs.")
    op.add_argument("--from-ted", type=Path, help="Restore files from a populated ted_consensus_1.0 installation.")
    op.add_argument("--pack", type=Path, help="Create a new offline tar archive of the verified files.")
    p.add_argument("--module", choices=["merizo","chainsaw","unidoc"], help="Verify or restore only this module.")
    args = p.parse_args()
    if args.pack and args.module:
        p.error("--pack creates a complete bundle. Omit --module.")
    manifest = json.loads((ROOT/"assets/manifest.json").read_text())
    online = args.download or args.download_weights or args.download_native
    urls = {CHAINSAW_STRIDE: CHAINSAW_STRIDE_URL}
    if args.download or args.download_weights:
        weight_urls = json.loads((ROOT/"assets/weight-urls.json").read_text())
        if weight_urls.keys() != {name for name in manifest if name.endswith(".pt")}:
            raise ValueError("Weight URLs and checksums list different files.")
        urls.update(weight_urls)
    checked = []
    with ExitStack() as resources:
        archive = resources.enter_context(tarfile.open(args.bundle, "r:*")) if args.bundle else None
        unidoc_archive = None
        for name, expected in manifest.items():
            path = Path(name)
            module = path.parts[1]
            if args.module and module != args.module:
                continue
            target = ROOT/path
            executable = target.suffix != ".pt"
            if (args.download_weights and executable) or (args.download_native and not executable):
                continue
            if not target.is_file() or checksum(target) != expected:
                if archive:
                    stream = member_stream(archive, name)
                elif online:
                    if name in UNIDOC_MEMBERS:
                        if unidoc_archive is None:
                            unidoc_archive = open_unidoc_archive(resources)
                        stream = member_stream(unidoc_archive, UNIDOC_MEMBERS[name])
                    else:
                        stream = download(urls[name])
                elif args.from_ted:
                    stream = (args.from_ted/"programs"/module/Path(*path.parts[3:])).open("rb")
                else:
                    raise SystemExit(f"Missing or mismatched asset: {name}. Use --download, --bundle or --from-ted.")
                with stream:
                    restore(stream, target, expected, executable)
            if executable and not target.stat().st_mode & 0o111:
                if online or args.bundle or args.from_ted:
                    target.chmod(0o755)
                else:
                    raise SystemExit(f"Missing executable permission: {name}")
            checked.append(name)
    if args.pack:
        with tarfile.open(args.pack, "x") as bundle:
            for name in checked:
                bundle.add(ROOT/name, arcname=name, recursive=False)
    kind = "model weights" if args.download_weights else "runtime assets"
    if args.download_native:
        kind = "native programs"
    print(f"Verified {len(checked)} {kind}.")

if __name__ == "__main__":
    main()
