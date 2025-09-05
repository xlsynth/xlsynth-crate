#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Ensure rustup environment is loaded if present
[ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"

echo "==> Installing xlsynth DSO"
python3 download_release.py -p ubuntu2004 -o xlsynth_tools -d
ls xlsynth_tools/*.so
mv -iv xlsynth_tools/*.so /usr/lib/

echo "==> Running ldconfig"
ldconfig

echo "==> Setting up environment variables"
export XLSYNTH_TOOLS="$PWD/xlsynth_tools"
export DSLX_STDLIB_PATH="$XLSYNTH_TOOLS/xls/dslx/stdlib"
export SLANG_PATH="$PWD/slang"
export PATH="$PATH:$PWD"
export XLS_DSO_PATH=$(ls /usr/lib/libxls*.so)

[ -f "$XLS_DSO_PATH" ] && echo "DSO found OK"

# Idempotently persist environment variables for later interactive use
ensure_line_in_bashrc() {
line="$1"
grep -qxF "$line" ~/.bashrc || echo "$line" >> ~/.bashrc
}

ensure_line_in_bashrc "export XLSYNTH_TOOLS=\"$XLSYNTH_TOOLS\""
ensure_line_in_bashrc "export DSLX_STDLIB_PATH=\"$DSLX_STDLIB_PATH\""
ensure_line_in_bashrc "export SLANG_PATH=\"$SLANG_PATH\""
ensure_line_in_bashrc "export XLS_DSO_PATH=\"$XLS_DSO_PATH\""
ensure_line_in_bashrc "export PATH=\"$PATH:$PWD\""

pre-commit install
# skip rustfmt for the moment as it's having an issue
SKIP=rustfmt pre-commit run --all-files

echo "==> Prefetching all Cargo dependencies"
cargo fetch --quiet

echo "==> Building fuzz target (zero-second run to ensure it builds)"
cd xlsynth-g8r && cargo fuzz run fuzz_gatify --max_seconds=0 && cd ..

echo "==> Pre-building workspace to run all build.rs scripts"
cargo build --workspace --all-targets --features=with-boolector-system --jobs $(nproc)

echo "==> Going offline (network locked)"
export CARGO_NET_OFFLINE=true
ensure_line_in_bashrc 'export CARGO_NET_OFFLINE=true'

echo "==> Showing bashrc"
cat ~/.bashrc

echo "✅ Maintenance complete — you can now run 'cargo test --workspace'"
