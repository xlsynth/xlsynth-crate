#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if dpkg -s rustc >/dev/null 2>&1 || dpkg -s cargo >/dev/null 2>&1; then
sudo apt-get remove -y rustc cargo || true
fi
sudo apt update && sudo apt upgrade -y
sudo apt install curl build-essential gcc make -y
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain nightly
source "$HOME/.cargo/env"

echo "==> Installing Rust nightly + clippy"
rustup toolchain install nightly --profile minimal
rustup component add clippy rustfmt --toolchain nightly
rustup override set nightly

echo "==> Installing system prerequisites"
sudo apt-get update -y
sudo apt-get install -y wget curl unzip gnupg cmake build-essential python3-pip pkg-config valgrind iverilog

echo "==> Installing LLVM 18 libc++/libc++abi (matches CI)"
curl -fsSL https://apt.llvm.org/llvm.sh -o /tmp/llvm.sh
chmod +x /tmp/llvm.sh
sudo /tmp/llvm.sh 18
sudo apt-get install -y libc++-18-dev libc++abi-18-dev

echo "==> Installing protoc 29.1"
PROTOC_ZIP=protoc-29.1-linux-x86_64.zip
curl -L -o ${PROTOC_ZIP} https://github.com/protocolbuffers/protobuf/releases/download/v29.1/${PROTOC_ZIP}
unzip -q ${PROTOC_ZIP} -d /tmp/protoc
sudo mv /tmp/protoc/bin/protoc /usr/local/bin/
rm -rf /tmp/protoc "${PROTOC_ZIP}"
protoc --version

echo "==> Installing Python deps and fetching XLSynth tools"
pip3 install --upgrade pip
pip3 install -r requirements.txt
python3 download_release.py -p ubuntu2004 -o xlsynth_tools -d

echo "==> Downloading Slang binary"
curl -L -o slang https://github.com/xlsynth/slang-rs/releases/download/ci/slang-rocky8
chmod +x slang

echo "==> Installing xlsynth DSO"
ls xlsynth_tools/*.so
mv -iv xlsynth_tools/*.so /usr/lib/

echo "==> Installing boolector DSO"
wget -O /usr/lib/libboolector.so https://github.com/xlsynth/boolector-build/releases/download/boolector-debian10-171b2783200bf9f7636f3e595587ee822a0a6d07/libboolector-debian10.so

echo "==> Running ldconfig"
ldconfig

echo "==> Setting up environment variables"
export XLSYNTH_TOOLS="$PWD/xlsynth_tools"
export DSLX_STDLIB_PATH="$XLSYNTH_TOOLS/xls/dslx/stdlib"
export SLANG_PATH="$PWD/slang"
export PATH="$PATH:$PWD"
export XLS_DSO_PATH=$(ls /usr/lib/libxls*.so)

[ -f "$XLS_DSO_PATH" ] && echo "DSO found OK"

pip3 install pre-commit
pre-commit install
# skip rustfmt for the moment as it's having an issue
SKIP=rustfmt pre-commit run --all-files

# Persist them for later interactive use
echo "export XLSYNTH_TOOLS=\"$XLSYNTH_TOOLS\""      >> ~/.bashrc
echo "export DSLX_STDLIB_PATH=\"$DSLX_STDLIB_PATH\"" >> ~/.bashrc
echo "export SLANG_PATH=\"$SLANG_PATH\""            >> ~/.bashrc
echo "export XLS_DSO_PATH=\"$XLS_DSO_PATH\""        >> ~/.bashrc
echo "export PATH=\"$PATH:$PWD\""                   >> ~/.bashrc

echo "==> Prefetching all Cargo dependencies"
cargo fetch --quiet

cargo install cargo-fuzz
cd xlsynth-g8r && cargo fuzz run fuzz_gatify --max_seconds=0 && cd ..

echo "==> Pre-building workspace to run all build.rs scripts"
cargo build --workspace --all-targets --features=with-boolector-system --jobs $(nproc)

echo "==> Going offline (network locked)"
export CARGO_NET_OFFLINE=true
echo 'export CARGO_NET_OFFLINE=true' >> ~/.bashrc

echo "==> Showing bashrc"
cat ~/.bashrc

echo "✅ Setup complete — you can now run 'cargo test --workspace'"
