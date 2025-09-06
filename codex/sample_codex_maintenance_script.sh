#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Ensure rustup environment is loaded if present
[ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"

echo "==> Installing xlsynth DSO"
# Note: we use system python for this as we installed the python3-requests package earlier
/usr/bin/python3 download_release.py -p ubuntu2004 -o xlsynth_tools -d
ls xlsynth_tools/*.so
mv -iv xlsynth_tools/*.so /usr/lib/

echo "==> Running ldconfig"
ldconfig

echo "==> Setting up environment variables"
export XLSYNTH_TOOLS="$PWD/xlsynth_tools"
export DSLX_STDLIB_PATH="$XLSYNTH_TOOLS/xls/dslx/stdlib"
export SLANG_PATH="/usr/local/bin/slang"
export PATH="$PATH:$PWD"
export XLS_DSO_PATH=$(ls /usr/lib/libxls*.so)

[ -f "$XLS_DSO_PATH" ] && echo "DSO found OK"

# Note: codex docs say we should append to bashrc for commands
# to pick up our env vars
# https://developers.openai.com/codex/cloud/environments#manual-setup
echo "==> Persisting environment variables to ~/.bashrc"
BASHRC="$HOME/.bashrc"
touch "$BASHRC"

append_if_missing() { grep -qxF "$1" "$BASHRC" || echo "$1" >> "$BASHRC"; }

append_if_missing "export XLSYNTH_TOOLS=$(printf '%q' "$XLSYNTH_TOOLS")"
append_if_missing "export DSLX_STDLIB_PATH=$(printf '%q' "$DSLX_STDLIB_PATH")"
append_if_missing "export SLANG_PATH=$(printf '%q' "$SLANG_PATH")"
append_if_missing "export XLS_DSO_PATH=$(printf '%q' "$XLS_DSO_PATH")"
append_if_missing "export PATH=\$PATH:$(printf '%q' "$PWD")"

pre-commit install

echo "==> Running pre-commit"
pre-commit run --all-files

echo "==> Prefetching all Cargo dependencies"
cargo fetch --quiet

echo "==> Building fuzz target (zero-second run to ensure it builds)"
cd xlsynth-g8r && cargo fuzz run fuzz_gatify --max_seconds=0 && cd ..

echo "==> Pre-building workspace to run all build.rs scripts"
cargo build --workspace --all-targets --features=with-boolector-system --jobs $(nproc)

echo "==> Going offline (network locked)"
export CARGO_NET_OFFLINE=true
append_if_missing "export CARGO_NET_OFFLINE=true"

echo "✅ Maintenance complete — you can now run 'cargo test --workspace'"
