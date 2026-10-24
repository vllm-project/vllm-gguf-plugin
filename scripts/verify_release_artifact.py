#!/usr/bin/env python3
"""Verify that release archives carry the required license materials."""

from __future__ import annotations

import argparse
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LICENSE_FILES = {
    "LICENSE": (ROOT / "LICENSE").read_bytes(),
    "LICENSES/llama.cpp-MIT.txt": (ROOT / "LICENSES/llama.cpp-MIT.txt").read_bytes(),
    "THIRD_PARTY_NOTICES.md": (ROOT / "THIRD_PARTY_NOTICES.md").read_bytes(),
}
METADATA_LINES = {
    b"License-Expression: Apache-2.0",
    b"License-File: LICENSE",
    b"License-File: LICENSES/llama.cpp-MIT.txt",
    b"License-File: THIRD_PARTY_NOTICES.md",
}


def _assert_metadata(content: bytes, artifact: Path) -> None:
    missing = sorted(line.decode() for line in METADATA_LINES if line not in content)
    if missing:
        raise RuntimeError(f"{artifact}: missing metadata entries: {missing}")


def _verify_sdist(artifact: Path) -> None:
    with tarfile.open(artifact, "r:gz") as archive:
        names = archive.getnames()
        roots = {name.split("/", 1)[0] for name in names if "/" in name}
        if len(roots) != 1:
            raise RuntimeError(f"{artifact}: expected one archive root, found {roots}")
        prefix = roots.pop()

        for relative_path, expected in LICENSE_FILES.items():
            member = archive.extractfile(f"{prefix}/{relative_path}")
            if member is None or member.read() != expected:
                raise RuntimeError(f"{artifact}: invalid {relative_path}")

        upstream_license = archive.extractfile(
            f"{prefix}/third_party/llama.cpp/LICENSE"
        )
        expected_upstream = LICENSE_FILES["LICENSES/llama.cpp-MIT.txt"]
        if upstream_license is None or upstream_license.read() != expected_upstream:
            raise RuntimeError(f"{artifact}: invalid upstream llama.cpp LICENSE")

        required_source = f"{prefix}/third_party/llama.cpp/ggml/src/ggml-cuda/mmq.cu"
        if required_source not in names:
            raise RuntimeError(f"{artifact}: missing selected llama.cpp sources")

        metadata = archive.extractfile(f"{prefix}/PKG-INFO")
        if metadata is None:
            raise RuntimeError(f"{artifact}: missing PKG-INFO")
        _assert_metadata(metadata.read(), artifact)


def _one_matching_name(names: list[str], suffix: str, artifact: Path) -> str:
    matches = [name for name in names if name.endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(
            f"{artifact}: expected one member ending in {suffix!r}, found {matches}"
        )
    return matches[0]


def _verify_wheel(artifact: Path) -> None:
    with zipfile.ZipFile(artifact) as archive:
        names = archive.namelist()
        for relative_path, expected in LICENSE_FILES.items():
            member = _one_matching_name(
                names, f".dist-info/licenses/{relative_path}", artifact
            )
            if archive.read(member) != expected:
                raise RuntimeError(f"{artifact}: invalid {relative_path}")

        metadata_name = _one_matching_name(names, ".dist-info/METADATA", artifact)
        _assert_metadata(archive.read(metadata_name), artifact)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts", nargs="+", type=Path)
    args = parser.parse_args()

    for artifact in args.artifacts:
        if artifact.name.endswith(".tar.gz"):
            _verify_sdist(artifact)
        elif artifact.suffix == ".whl":
            _verify_wheel(artifact)
        else:
            raise RuntimeError(f"Unsupported release artifact: {artifact}")
        print(f"verified license materials in {artifact}")


if __name__ == "__main__":
    main()
