# SPDX-License-Identifier: Apache-2.0

"""Downloads and runs the rustup installer with retry-friendly CI diagnostics."""

import argparse
import os
import subprocess
import sys
import tempfile
import time
import urllib.request


DEFAULT_INSTALLER_URL = "https://sh.rustup.rs"


def download_with_retry(url, destination, max_attempts, initial_delay_seconds):
    """Downloads the rustup installer script with retries and backoff."""
    delay_seconds = initial_delay_seconds
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "xlsynth-ci-rustup-bootstrap/1.0"},
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = response.read()
            with open(destination, "wb") as f:
                f.write(payload)
            return
        except Exception as e:
            last_error = e
            print(
                "Attempt {} of {} failed while downloading {}: {}".format(
                    attempt, max_attempts, url, e
                ),
                file=sys.stderr,
            )
            if attempt == max_attempts:
                break
            time.sleep(delay_seconds)
            delay_seconds *= 2
    raise RuntimeError(
        "Failed to download rustup installer from {} after {} attempts: {}".format(
            url, max_attempts, last_error
        )
    )


def build_install_command(installer_path, profile, default_toolchain):
    """Builds the rustup installer command line."""
    command = ["sh", installer_path, "-y"]
    if profile:
        command.extend(["--profile", profile])
    if default_toolchain:
        command.extend(["--default-toolchain", default_toolchain])
    return command


def ensure_expected_binaries_exist():
    """Verifies rustup created the expected Cargo tool binaries."""
    cargo_bin = os.path.expanduser("~/.cargo/bin/cargo")
    rustc_bin = os.path.expanduser("~/.cargo/bin/rustc")
    missing = [path for path in (cargo_bin, rustc_bin) if not os.path.exists(path)]
    if missing:
        raise RuntimeError(
            "rustup installer completed but expected binaries were missing: {}".format(
                ", ".join(missing)
            )
        )


def main():
    parser = argparse.ArgumentParser(
        description="Downloads and runs rustup with retry-friendly CI diagnostics."
    )
    parser.add_argument(
        "--installer-url",
        default=DEFAULT_INSTALLER_URL,
        help="URL to download the rustup shell installer from.",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="Optional rustup profile to install, e.g. minimal.",
    )
    parser.add_argument(
        "--default-toolchain",
        default=None,
        help="Optional default toolchain to install, e.g. nightly.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=5,
        help="Maximum number of installer download attempts before failing.",
    )
    parser.add_argument(
        "--initial-delay-seconds",
        type=int,
        default=2,
        help="Initial retry delay in seconds; later attempts use exponential backoff.",
    )
    args = parser.parse_args()

    if args.max_attempts < 1:
        raise ValueError("--max-attempts must be at least 1")
    if args.initial_delay_seconds < 0:
        raise ValueError("--initial-delay-seconds must be non-negative")

    installer_path = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="rustup-init-", suffix=".sh", delete=False
        ) as installer_file:
            installer_path = installer_file.name
        print(
            "Downloading rustup installer from {} with up to {} attempts".format(
                args.installer_url, args.max_attempts
            )
        )
        download_with_retry(
            args.installer_url,
            installer_path,
            args.max_attempts,
            args.initial_delay_seconds,
        )
        os.chmod(installer_path, 0o755)
        install_command = build_install_command(
            installer_path, args.profile, args.default_toolchain
        )
        print("Running rustup installer: {}".format(" ".join(install_command)))
        subprocess.run(install_command, check=True)
        ensure_expected_binaries_exist()
    except subprocess.CalledProcessError as e:
        print(
            "rustup installer exited with code {} while running {}".format(
                e.returncode, " ".join(e.cmd)
            ),
            file=sys.stderr,
        )
        raise
    finally:
        if installer_path is not None and os.path.exists(installer_path):
            os.remove(installer_path)


if __name__ == "__main__":
    main()
