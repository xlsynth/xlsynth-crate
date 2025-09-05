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

cargo install cargo-nextest

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

echo "==> Installing Python deps for scripts"
pip3 install --upgrade pip
pip3 install -r requirements.txt
pip3 install pre-commit

echo "==> Downloading Slang binary"
curl -L -o /usr/local/bin/slang https://github.com/xlsynth/slang-rs/releases/download/ci/slang-rocky8
chmod +x /usr/local/bin/slang

echo "==> Installing boolector DSO"
wget -O /usr/lib/libboolector.so https://github.com/xlsynth/boolector-build/releases/download/boolector-debian10-171b2783200bf9f7636f3e595587ee822a0a6d07/libboolector-debian10.so

echo "✅ Base setup complete — run 'bash sample_codex_maintenance_script.sh' per job"
