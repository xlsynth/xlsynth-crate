# SPDX-License-Identifier: Apache-2.0

import pathlib
import re
import sys


def main() -> int:
    build_rs = pathlib.Path("xlsynth-sys/build.rs")
    try:
        contents = build_rs.read_text()
    except Exception as exc:  # defensive: fail clearly for CI logs
        print(f"error: failed to read {build_rs}: {exc}", file=sys.stderr)
        return 1

    match = re.search(
        r'const\s+RELEASE_LIB_VERSION_TAG:\s*&str\s*=\s*"([^"]+)"', contents
    )
    if not match:
        print(
            "error: could not find RELEASE_LIB_VERSION_TAG const in xlsynth-sys/build.rs",
            file=sys.stderr,
        )
        return 1

    tag = match.group(1)
    # Print only the tag to stdout so shell capture is simple and safe.
    print(tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
