#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

mode="full"
if [[ "$#" -eq 2 && "$1" == "--mode" ]]; then
  mode="$2"
elif [[ "$#" -ne 0 ]]; then
  echo "Usage: $0 [--mode full|build-only]" >&2
  exit 2
fi
case "${mode}" in
  full | build-only) ;;
  *)
    echo "Usage: $0 [--mode full|build-only]" >&2
    exit 2
    ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

# Source explicitly: non-interactive shells need not read any shell rc file.
source "$HOME/.cargo/env"
source "$HOME/.xlsynth_container_env.sh"
export CARGO_NET_OFFLINE=true

# --frozen checks both network independence and the dependency resolution saved
# during preparation. The default optimizer/prover paths require Bitwuzla.
if [[ "${mode}" == "build-only" ]]; then
  # Preparation intentionally did not compile the main workspace in this mode,
  # so this is a real network-disabled build rather than a cached no-op.
  cargo build --workspace --frozen --features with-bitwuzla-system
  exit 0
fi

# Nextest does not run doctests, so exercise those as well in full mode.
cargo check --workspace --all-targets --frozen --features with-bitwuzla-system
cargo nextest run --workspace --frozen --features with-bitwuzla-system --no-fail-fast --test-threads "${CARGO_BUILD_JOBS:-4}"
cargo test --workspace --doc --frozen --features with-bitwuzla-system
SKIP=no-commit-to-branch pre-commit run --all-files
