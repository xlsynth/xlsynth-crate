#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [[ "$#" -ne 0 ]]; then
  echo "Usage: $0" >&2
  exit 2
fi

if [[ "$(uname -m)" != "x86_64" ]]; then
  echo "This Docker image requires Linux x86-64; build it with --platform linux/amd64." >&2
  exit 1
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

# Tool installation is an online build phase, even if the invoking shell
# inherited an offline Cargo environment.
unset CARGO_NET_OFFLINE

as_root() {
  if [[ "$EUID" -eq 0 ]]; then
    "$@"
  else
    sudo "$@"
  fi
}

echo "==> Installing system prerequisites"
as_root bash scripts/ci_apt_retry.sh update
as_root bash scripts/ci_apt_retry.sh install --no-install-recommends \
  ca-certificates curl unzip git cmake build-essential python3 python3-requests \
  python3-venv pre-commit pkg-config valgrind iverilog libc++-18-dev libc++abi-18-dev

if dpkg -s rustc >/dev/null 2>&1 || dpkg -s cargo >/dev/null 2>&1; then
  as_root apt-get remove -y rustc cargo
fi
rust_toolchain="${RUSTUP_TOOLCHAIN:-nightly}"
python3 scripts/ci_install_rustup.py --profile minimal --default-toolchain "${rust_toolchain}"
source "$HOME/.cargo/env"

echo "==> Installing Rust ${rust_toolchain} + clippy"
rustup component add clippy rustfmt --toolchain "${rust_toolchain}"
rustup override set "${rust_toolchain}"

echo "==> Installing cargo extensions"
cargo install --locked --version 0.9.143 cargo-nextest
cargo install --locked --version 0.13.2 cargo-fuzz

downloads_dir="$(mktemp -d)"
trap 'rm -rf "${downloads_dir}"' EXIT

echo "==> Installing protoc 29.1"
curl --fail --location --retry 5 -o "${downloads_dir}/protoc.zip" \
  https://github.com/protocolbuffers/protobuf/releases/download/v29.1/protoc-29.1-linux-x86_64.zip
unzip -q "${downloads_dir}/protoc.zip" -d "${downloads_dir}/protoc"
as_root install -m 0755 "${downloads_dir}/protoc/bin/protoc" /usr/local/bin/protoc
protoc --version

echo "==> Downloading Slang binary"
slang_sha256="$(tr -d '\n' < scripts/slang_rocky8.sha256)"
python3 scripts/download_and_verify.py elf "${downloads_dir}/slang" \
  https://api.github.com/repos/xlsynth/slang-rs/releases/assets/220397578 \
  --sha256 "${slang_sha256}"
as_root install -m 0755 "${downloads_dir}/slang" /usr/local/bin/slang

echo "==> Installing boolector DSO"
python3 scripts/download_and_verify.py elf "${downloads_dir}/libboolector.so" \
  https://github.com/xlsynth/boolector-build/releases/download/boolector-debian10-171b2783200bf9f7636f3e595587ee822a0a6d07/libboolector-debian10.so
as_root install -m 0644 "${downloads_dir}/libboolector.so" /usr/lib/libboolector.so

echo "==> Installing Bitwuzla DSOs (matches CI)"
bitwuzla_release="https://github.com/xlsynth/boolector-build/releases/download/bitwuzla-binaries-b29041fbbe6318cb4c19a6e11c7616efc4cb4d32"
for library in bitwuzla bitwuzlabb bitwuzlabv bitwuzlals cadical; do
  artifact="lib${library}-rocky8.so"
  python3 scripts/download_and_verify.py elf "${downloads_dir}/${artifact}" \
    "${bitwuzla_release}/${artifact}"
  as_root install -m 0644 "${downloads_dir}/${artifact}" "/usr/lib/lib${library}.so"
done
as_root ldconfig

echo "✅ Tool installation complete"
