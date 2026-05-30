#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import sys
import time
import urllib.error

from crates_io_index import crate_version_is_published


POLL_INTERVAL_SECONDS = 15
TIMEOUT_TOTAL_SECONDS = 5 * 60
MAX_DOTS_BEFORE_NEWLINE = 60


def check_crate_version(crate_name, version_wanted):
    """Return whether the crate version is visible in the sparse index."""
    return crate_version_is_published(crate_name, version_wanted)


def check_crate_version_for_polling(crate_name, version_wanted):
    """Return version visibility while retrying transient post-publish failures."""
    try:
        return check_crate_version(crate_name, version_wanted)
    except urllib.error.HTTPError as error:
        if 500 <= error.code < 600:
            print(
                "\n[Sparse index HTTP error: {}]".format(error.code),
                end="",
                flush=True,
            )
            return False
        raise
    except (urllib.error.URLError, TimeoutError) as error:
        print(
            "\n[Sparse index request error: {}]".format(error),
            end="",
            flush=True,
        )
        return False


def wait_until_version_seen(
    crate_name,
    version_wanted,
    poll_interval_seconds=POLL_INTERVAL_SECONDS,
    timeout_total_seconds=TIMEOUT_TOTAL_SECONDS,
):
    """Return whether a crate version becomes visible before the timeout."""
    print(
        f"Waiting for {crate_name} v{version_wanted} to appear in the crates.io sparse index (polling every {poll_interval_seconds}s, timeout: {timeout_total_seconds}s)...",
        end="",
        flush=True,
    )

    start_time = time.time()
    iteration_count = 0
    while True:
        if check_crate_version_for_polling(crate_name, version_wanted):
            print("\nVersion found!")
            return True

        current_time = time.time()
        if current_time - start_time > timeout_total_seconds:
            print(
                f"\nTimeout reached after {int(current_time - start_time)} seconds. Version {version_wanted} of {crate_name} not found."
            )
            return False

        time.sleep(poll_interval_seconds)
        print(".", end="", flush=True)
        iteration_count += 1
        if iteration_count % MAX_DOTS_BEFORE_NEWLINE == 0:
            print(f" [{int(time.time() - start_time)}s elapsed]", end="", flush=True)
            print("\n", end="", flush=True)


def main():
    if len(sys.argv) != 3:
        print(
            "Usage: python3 scripts/sleep_until_version_seen.py <crate-name> <version-wanted>"
        )
        sys.exit(2)

    crate_name = sys.argv[1]
    version_wanted = sys.argv[2]

    try:
        if not wait_until_version_seen(crate_name, version_wanted):
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nPolling interrupted by user.")
        sys.exit(3)


if __name__ == "__main__":
    main()
