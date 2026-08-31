#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Install pinned, checksum-verified Linux x64 Yosys/ABC for codegen CI tests."""

import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile
import urllib.request


URL = (
    "https://github.com/YosysHQ/oss-cad-suite-build/releases/download/"
    "2026-08-29/oss-cad-suite-linux-x64-20260829.tgz"
)
SHA256 = "dd687a4694ac67b2ee31841473efbd1f49131b8b707d07213e527bf295105466"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        archive = Path(temporary) / "suite.tgz"
        with urllib.request.urlopen(URL, timeout=120) as response, archive.open(
            "wb"
        ) as output:
            shutil.copyfileobj(response, output)
        digest = hashlib.sha256()
        with archive.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != SHA256:
            raise RuntimeError("OSS CAD Suite archive checksum mismatch")
        # Extract only after verifying the pinned upstream archive, including its
        # bundled runtime and symlinks needed by the Yosys/ABC launcher scripts.
        subprocess.check_call(
            ["tar", "-xzf", str(archive), "-C", str(args.destination)]
        )


if __name__ == "__main__":
    main()
