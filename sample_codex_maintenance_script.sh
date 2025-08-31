#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Ensure environment variables and cargo are available
source "$HOME/.cargo/env"
source ~/.bashrc

pre-commit install
# skip rustfmt for the moment as it's having an issue
SKIP=rustfmt pre-commit run --all-files

echo "==> Prefetching all Cargo dependencies"
cargo fetch --quiet

cd xlsynth-g8r && cargo fuzz run fuzz_gatify --max_seconds=0 && cd ..

echo "==> Pre-building workspace to run all build.rs scripts"
cargo build --workspace --all-targets --features=with-boolector-system --jobs $(nproc)

echo "==> Going offline (network locked)"
export CARGO_NET_OFFLINE=true
echo 'export CARGO_NET_OFFLINE=true' >> ~/.bashrc

echo "==> Showing bashrc"
cat ~/.bashrc

echo "✅ Maintenance complete — you can now run 'cargo test --workspace'"
