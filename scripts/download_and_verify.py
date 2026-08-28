#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Downloads CI binary artifacts and rejects HTML/error payloads early.

This exists because several workflows need the same "download with retry, then
verify the artifact is actually the expected binary format" behavior, and
keeping that logic inline in YAML made it both duplicated and shell-fragile.
Centralizing it in Python also avoids depending on runner-specific curl features
such as `--retry-all-errors`, which are missing on older images like Rocky 8.
"""

import argparse
import hashlib
import os
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from pathlib import Path
from typing import Optional

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


def normalize_sha256(value: str) -> str:
    if len(value) != 64 or not all(c in "0123456789abcdefABCDEF" for c in value):
        raise ValueError("SHA-256 must be exactly 64 hexadecimal characters")
    return value.lower()


def parse_sha256_argument(value: str) -> str:
    try:
        return normalize_sha256(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("elf", "dylib", "tar-gz", "zip"))
    parser.add_argument("output")
    parser.add_argument("url")
    checksum_group = parser.add_mutually_exclusive_group()
    checksum_group.add_argument("--sha256", type=parse_sha256_argument)
    checksum_group.add_argument("--sha256-url")
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=int, default=60)
    return parser.parse_args()


def build_request(url: str) -> urllib.request.Request:
    headers = {
        "Accept": "application/octet-stream",
        "User-Agent": "xlsynth-ci-artifact-fetcher",
    }
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


def download_and_verify_with_retry(
    kind: str,
    url: str,
    destination: Path,
    attempts: int,
    timeout_seconds: int,
    sha256: Optional[str] = None,
    sha256_url: Optional[str] = None,
) -> None:
    if sha256 is not None and sha256_url is not None:
        raise ValueError("--sha256 and --sha256-url are mutually exclusive")
    expected_sha256 = normalize_sha256(sha256) if sha256 is not None else None

    destination.parent.mkdir(parents=True, exist_ok=True)
    delay_seconds = 2
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            download_with_retry(
                url, destination, attempts=1, timeout_seconds=timeout_seconds
            )
            validation_error = validate_artifact(destination, kind)
            if validation_error is None:
                if expected_sha256 is not None:
                    validation_error = validate_sha256(destination, expected_sha256)
                else:
                    validation_error = validate_sha256_url(
                        destination, sha256_url, timeout_seconds
                    )
            if validation_error is None:
                return
            if expected_sha256 is not None or sha256_url is not None:
                destination.unlink()
            last_error = RuntimeError(validation_error)
        except (urllib.error.URLError, OSError, RuntimeError) as exc:
            last_error = exc

        if attempt == attempts:
            break
        print(
            f"Attempt {attempt} produced an invalid artifact for {url}: {last_error}. "
            f"Retrying in {delay_seconds} seconds...",
            file=sys.stderr,
        )
        time.sleep(delay_seconds)
        delay_seconds *= 2

    assert last_error is not None
    raise last_error


def detect_binary_kind(path: Path) -> str:
    with path.open("rb") as f:
        magic = f.read(4)
    if magic == ELF_MAGIC:
        return "elf"
    if magic in MACH_O_MAGICS:
        return "dylib"
    return "unknown"


def validate_tar_gz(path: Path) -> str:
    try:
        with tarfile.open(str(path), "r:gz") as archive:
            archive.getmembers()
        return ""
    except (tarfile.TarError, OSError) as exc:
        return str(exc)


def validate_zip(path: Path) -> str:
    try:
        with zipfile.ZipFile(str(path), "r") as archive:
            bad_member = archive.testzip()
            if bad_member is not None:
                return f"CRC validation failed for archive member {bad_member}"
        return ""
    except (zipfile.BadZipFile, zlib.error, OSError) as exc:
        return str(exc)


def describe_binary(path: Path) -> str:
    try:
        import subprocess

        result = subprocess.run(
            ["file", "-b", str(path)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
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
        return "ELF binary"
    if kind == "dylib":
        return "Mach-O dynamic library"
    if kind == "zip":
        return "ZIP archive"
    return "gzip-compressed tar archive"


def validate_artifact(path: Path, kind: str):
    description = describe_binary(path)
    print(f"{path}: {description}")

    if kind == "tar-gz":
        error = validate_tar_gz(path)
        if not error:
            return None
        return (
            f"Expected {expected_description(kind)}, got '{description}' "
            f"from {path}: {error}"
        )
    if kind == "zip":
        error = validate_zip(path)
        if not error:
            return None
        return (
            f"Expected {expected_description(kind)}, got '{description}' "
            f"from {path}: {error}"
        )

    actual_kind = detect_binary_kind(path)
    if actual_kind == kind:
        return None
    return f"Expected {expected_description(kind)}, got '{description}' from {path}"


def parse_sha256_text(text: str) -> str:
    for token in text.split():
        if len(token) == 64 and all(c in "0123456789abcdefABCDEF" for c in token):
            return token.lower()
    raise ValueError("no SHA-256 digest found in checksum payload")


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_sha256(path: Path, expected: str):
    actual = sha256_file(path)
    if actual == expected:
        print(f"{path}: sha256 verified {actual}")
        return None
    return f"SHA-256 mismatch for {path}: expected {expected}, got {actual}"


def validate_sha256_url(path: Path, sha256_url: Optional[str], timeout_seconds: int):
    if not sha256_url:
        return None
    try:
        with urllib.request.urlopen(
            build_request(sha256_url), timeout=timeout_seconds
        ) as response:
            checksum_text = response.read().decode("utf-8", errors="replace")
        expected = parse_sha256_text(checksum_text)
    except (urllib.error.URLError, OSError, UnicodeError, ValueError) as exc:
        return f"could not fetch or parse checksum {sha256_url}: {exc}"

    return validate_sha256(path, expected)


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    try:
        download_and_verify_with_retry(
            args.kind,
            args.url,
            output,
            args.attempts,
            args.timeout_seconds,
            sha256=args.sha256,
            sha256_url=args.sha256_url,
        )
    except (urllib.error.URLError, OSError, RuntimeError, ValueError) as exc:
        print(
            f"Failed to download valid {expected_description(args.kind)} "
            f"from {args.url}: {exc}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
