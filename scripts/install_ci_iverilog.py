#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Install checksum-verified rocky8 iverilog and vvp artifacts for CI."""

from pathlib import Path
import subprocess
import sys
import tempfile


BASE_URL = (
    "https://github.com/xlsynth/boolector-build/releases/download/"
    "iverilog-binaries-ea26587b5ef485f2ca82a3e4364e58ec3307240f"
)
DOWNLOAD_AND_VERIFY = Path(__file__).resolve().parent / "download_and_verify.py"


def download(kind, destination, name):
    """Downloads one verified release artifact into the temporary directory."""
    subprocess.check_call(
        [
            sys.executable,
            str(DOWNLOAD_AND_VERIFY),
            kind,
            str(destination),
            "{}/{}".format(BASE_URL, name),
            "--sha256-url",
            "{}/{}.sha256".format(BASE_URL, name),
        ]
    )


def main():
    with tempfile.TemporaryDirectory(prefix="install_ci_iverilog_") as temporary:
        root = Path(temporary)
        compiler = root / "iverilog-rocky8"
        runtime = root / "vvp-rocky8"
        libraries = root / "ivl-rocky8.tar.gz"
        download("elf", compiler, compiler.name)
        download("elf", runtime, runtime.name)
        download("tar-gz", libraries, libraries.name)
        subprocess.check_call(["sudo", "mkdir", "-p", "/usr/local/bin", "/usr/local/lib"])
        subprocess.check_call(
            ["sudo", "install", "-m", "0755", str(compiler), "/usr/local/bin/iverilog"]
        )
        subprocess.check_call(
            ["sudo", "install", "-m", "0755", str(runtime), "/usr/local/bin/vvp"]
        )
        subprocess.check_call(
            ["sudo", "tar", "-xzf", str(libraries), "-C", "/usr/local/lib"]
        )
    subprocess.check_call(["iverilog", "-V"])
    subprocess.check_call(["vvp", "-V"])


if __name__ == "__main__":
    main()
