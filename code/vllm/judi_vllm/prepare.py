"""Prepare vLLM 0.8.3 from the active environment or the pinned official wheel."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
BASE = PACKAGE.parent
SOURCE = BASE / "vendor" / "vllm"
RUNTIME = BASE / ".runtime"
PIN = json.loads((BASE / "UPSTREAM.json").read_text())
WHEEL = PIN["wheel"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def installed_vllm():
    # Ignore our own generated metadata, including on a repeated invocation
    # whose PYTHONPATH already contains the local runtime.
    search_paths = [p for p in sys.path if not Path(p).resolve().is_relative_to(BASE)]
    for dist in importlib.metadata.distributions(path=search_paths):
        if dist.metadata.get("Name", "").lower() != "vllm":
            continue
        if dist.version != PIN["version"]:
            return None
        for name in dist.files or []:
            if str(name).endswith(".dist-info/METADATA"):
                return dist, Path(dist.locate_file(name)).resolve().parent
    return None


def prepare_installed(dist, metadata_dir: Path) -> dict:
    required = WHEEL["generated_files"]
    sources = {name: Path(dist.locate_file(name)).resolve() for name in required}
    missing = [name for name, path in sources.items() if not path.is_file()]
    if missing:
        raise RuntimeError(f"Installed vLLM 0.8.3 is missing CUDA runtime files: {missing}")
    # Copy only the generated files. The vendored upstream source stays intact.
    for name, source in sources.items():
        target = SOURCE / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    shutil.copytree(metadata_dir, RUNTIME / "vllm-0.8.3.dist-info", dirs_exist_ok=True)
    manifest = {
        "source_commit": PIN["commit"],
        "vllm_version": PIN["version"],
        "artifact_source": "installed_vllm",
        "installed_metadata": str(metadata_dir),
        "generated_files": {name: sha256(SOURCE / name) for name in sorted(required)},
    }
    (RUNTIME / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def wheel_path(requested: str | None) -> Path:
    if requested:
        path = Path(requested).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    cached = os.environ.get("VLLM_JUDI_WHEEL_CACHE")
    if cached:
        path = Path(cached).expanduser().resolve()
        if path.is_file():
            return path
    download_dir = BASE / ".downloads"
    download_dir.mkdir(exist_ok=True)
    path = download_dir / WHEEL["filename"]
    if not path.is_file() or sha256(path) != WHEEL["sha256"]:
        print(f"Downloading {WHEEL['filename']} ...", file=sys.stderr)
        with urllib.request.urlopen(WHEEL["url"]) as response, path.open("wb") as out:
            shutil.copyfileobj(response, out)
    return path


def prepare(requested: str | None = None, force: bool = False) -> dict:
    if not requested and not os.environ.get("VLLM_JUDI_WHEEL_CACHE"):
        installed = installed_vllm()
        if installed is not None:
            return prepare_installed(*installed)
    path = wheel_path(requested)
    actual = sha256(path)
    if actual != WHEEL["sha256"]:
        raise RuntimeError(f"Wheel SHA256 mismatch: {actual} != {WHEEL['sha256']}")
    with zipfile.ZipFile(path) as wheel:
        names = set(wheel.namelist())
        required = set(WHEEL["generated_files"])
        missing = sorted(required - names)
        if missing:
            raise RuntimeError(f"Pinned wheel is missing files: {missing}")
        if force:
            for name in required:
                target = SOURCE / name
                if target.exists():
                    target.unlink()
            shutil.rmtree(RUNTIME, ignore_errors=True)
        for name in required:
            target = SOURCE / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(wheel.read(name))
        dist_members = [
            name for name in names
            if name.startswith("vllm-0.8.3.dist-info/") and not name.endswith("/")
        ]
        dist_root = RUNTIME / "vllm-0.8.3.dist-info"
        dist_root.mkdir(parents=True, exist_ok=True)
        for name in dist_members:
            (RUNTIME / name).parent.mkdir(parents=True, exist_ok=True)
            (RUNTIME / name).write_bytes(wheel.read(name))
    manifest = {
        "source_commit": PIN["commit"],
        "vllm_version": PIN["version"],
        "wheel_sha256": actual,
        "wheel": WHEEL["filename"],
        "generated_files": {
            name: sha256(SOURCE / name) for name in sorted(required)
        },
    }
    RUNTIME.mkdir(exist_ok=True)
    (RUNTIME / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", help="local official vLLM 0.8.3 wheel")
    parser.add_argument("--force", action="store_true", help="replace existing generated files")
    parser.add_argument("--check", action="store_true", help="only verify the prepared runtime")
    args = parser.parse_args()
    if args.check:
        from .runtime import describe
        print(json.dumps(describe(), indent=2))
    else:
        print(json.dumps(prepare(args.wheel, args.force), indent=2))


if __name__ == "__main__":
    main()
