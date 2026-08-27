#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

# Source explicitly: non-interactive shells need not read any shell rc file.
source "$HOME/.cargo/env"
source "$HOME/.xlsynth_container_env.sh"
export CARGO_NET_OFFLINE=true

# --frozen checks both network independence and the dependency resolution saved
# during preparation. Nextest does not run doctests, so exercise those as well.
# The default optimizer/prover paths require Bitwuzla; keep the feature set
# identical across preparation, checking, and testing.
cargo check --workspace --all-targets --frozen --features with-bitwuzla-system
cargo nextest run --workspace --frozen --features with-bitwuzla-system --no-fail-fast --test-threads "${CARGO_BUILD_JOBS:-4}"
cargo test --workspace --doc --frozen --features with-bitwuzla-system
SKIP=no-commit-to-branch pre-commit run --all-files
