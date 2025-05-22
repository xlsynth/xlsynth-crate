#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import sys
import time
import requests
import json


def check_crate_version(crate_name, version_wanted):
    """
    Checks crates.io API for the specified version of a crate.
    Returns True if found, False otherwise.
    Prints brief error messages for API or JSON issues without exiting.
    """
    api_url = f"https://crates.io/api/v1/crates/{crate_name}"
    try:
        # Set a timeout for the request itself (e.g., 15 seconds)
        response = requests.get(api_url, timeout=15)
        # Raise an exception for HTTP errors (4xx or 5xx)
        response.raise_for_status()
        data = response.json()
        # The 'versions' key contains a list of version objects
        versions_data = data.get("versions", [])
        for v_info in versions_data:
            if v_info.get("num") == version_wanted:
                return True
    except requests.exceptions.Timeout:
        print("\n[API Request Timeout]", end="", flush=True)
    except requests.exceptions.HTTPError as e:
        # Log HTTP errors (like 404 if crate not found yet), but continue polling
        print(f"\n[API HTTP Error: {e.response.status_code}]", end="", flush=True)
    except requests.exceptions.RequestException as e:
        # Other request-related errors (DNS, connection, etc.)
        print(f"\n[API Request Error: {e}]", end="", flush=True)
    except json.JSONDecodeError:
        print("\n[API JSON Decode Error]", end="", flush=True)
    return False


def main():
    if len(sys.argv) != 3:
        print(
            "Usage: python3 sleep_until_version_seen.py <crate-name> <version-wanted>"
        )
        sys.exit(2)

    crate_name = sys.argv[1]
    version_wanted = sys.argv[2]

    # Polling parameters
    poll_interval_seconds = 15  # Check every 15 seconds
    timeout_total_seconds = 5 * 60  # Total timeout of 5 minutes

    print(
        f"Waiting for {crate_name} v{version_wanted} to appear on crates.io (polling every {poll_interval_seconds}s, timeout: {timeout_total_seconds}s)...",
        end="",
        flush=True,
    )

    start_time = time.time()
    iteration_count = 0
    max_dots_before_newline = 60  # Print a newline after this many dots

    try:
        while True:
            if check_crate_version(crate_name, version_wanted):
                print("\nVersion found!")
                sys.exit(0)

            current_time = time.time()
            if current_time - start_time > timeout_total_seconds:
                print(
                    f"\nTimeout reached after {int(current_time - start_time)} seconds. Version {version_wanted} of {crate_name} not found."
                )
                sys.exit(1)

            time.sleep(poll_interval_seconds)
            print(".", end="", flush=True)
            iteration_count += 1
            if iteration_count % max_dots_before_newline == 0:
                print(
                    f" [{int(time.time() - start_time)}s elapsed]", end="", flush=True
                )  # Optional: print elapsed time periodically
                print("\n", end="", flush=True)  # Start new line for dots

    except KeyboardInterrupt:
        print("\nPolling interrupted by user.")
        sys.exit(3)


if __name__ == "__main__":
    main()
