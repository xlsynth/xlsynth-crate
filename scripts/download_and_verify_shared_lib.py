#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Downloads solver shared libraries in CI and rejects HTML/error payloads early.

This exists because several workflows need the same "download with retry, then
verify the artifact is actually a shared library" behavior, and keeping that
logic inline in YAML made it both duplicated and shell-fragile. Centralizing it
in Python also avoids depending on runner-specific curl features such as
`--retry-all-errors`, which are missing on older images like Rocky 8.
"""

import argparse
import os
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ELF_MAGIC = b"\x7fELF"
MACH_O_MAGICS = {
    b"\xfe\xed\xfa\xce",
    b"\xfe\xed\xfa\xcf",
    b"\xce\xfa\xed\xfe",
    b"\xcf\xfa\xed\xfe",
    b"\xca\xfe\xba\xbe",
    b"\xbe\xba\xfe\xca",
    b"\xca\xfe\xba\xbf",
    b"\xbf\xba\xfe\xca",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("elf", "dylib"))
    parser.add_argument("output")
    parser.add_argument("url")
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=int, default=60)
    return parser.parse_args()


def build_request(url: str) -> urllib.request.Request:
    headers = {"User-Agent": "xlsynth-ci-shared-lib-fetcher"}
    gh_pat = os.getenv("GH_PAT")
    if gh_pat:
        headers["Authorization"] = f"token {gh_pat}"
    return urllib.request.Request(url, headers=headers)


def download_with_retry(
    url: str, destination: Path, attempts: int, timeout_seconds: int
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path_str = tempfile.mkstemp(
        prefix=f"{destination.name}.",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    os.close(fd)
    temp_path = Path(temp_path_str)
    delay_seconds = 2
    last_error = None  # type: Optional[Exception]
    try:
        for attempt in range(1, attempts + 1):
            try:
                with urllib.request.urlopen(
                    build_request(url),
                    timeout=timeout_seconds,
                ) as response, temp_path.open("wb") as out_file:
                    shutil.copyfileobj(response, out_file)
                temp_path.replace(destination)
                return
            except (urllib.error.URLError, OSError) as exc:
                last_error = exc
                if attempt == attempts:
                    break
                print(
                    f"Attempt {attempt} failed for {url}: {exc}. "
                    f"Retrying in {delay_seconds} seconds...",
                    file=sys.stderr,
                )
                time.sleep(delay_seconds)
                delay_seconds *= 2
        assert last_error is not None
        raise last_error
    finally:
        if temp_path.exists():
            temp_path.unlink()


def detect_binary_kind(path: Path) -> str:
    with path.open("rb") as f:
        magic = f.read(4)
    if magic == ELF_MAGIC:
        return "elf"
    if magic in MACH_O_MAGICS:
        return "dylib"
    return "unknown"


def describe_binary(path: Path) -> str:
    try:
        import subprocess

        result = subprocess.run(
            ["file", "-b", str(path)],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except OSError:
        pass

    with path.open("rb") as f:
        magic = f.read(8)
    return f"magic={magic.hex()}"


def expected_description(kind: str) -> str:
    if kind == "elf":
        return "ELF shared library"
    return "Mach-O dynamic library"


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    download_with_retry(args.url, output, args.attempts, args.timeout_seconds)

    description = describe_binary(output)
    print(f"{output}: {description}")

    actual_kind = detect_binary_kind(output)
    if actual_kind != args.kind:
        print(
            f"Expected {expected_description(args.kind)}, got '{description}' from {args.url}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
