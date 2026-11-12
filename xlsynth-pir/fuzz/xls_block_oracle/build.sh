#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

if (( $# == 0 )); then
  echo "Usage: bash build.sh <scratch-directory> [bazel build flags...]" >&2
  exit 2
fi

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$1"
work="$(cd "$1" && pwd)"
shift
tag="$(python3 "$here/../../../scripts/get_required_libxls_dso_and_tools_release_tag.py")"
source_dir="$work/xlsynth-$tag"

if [[ ! -d "$source_dir" ]]; then
  staging="$(mktemp -d "$work/unpack.XXXXXXXX")"
  trap 'rm -rf "$staging"' EXIT
  curl --fail --location --silent --show-error \
    "https://codeload.github.com/xlsynth/xlsynth/tar.gz/refs/tags/$tag" \
    --output "$staging/source.tar.gz"
  tar -xzf "$staging/source.tar.gz" -C "$staging"
  mv "$staging/xlsynth-${tag#v}" "$source_dir"
fi

mkdir -p "$source_dir/xls/pir_block_oracle"
cp "$here/BUILD.bazel" "$here/block_oracle.cc" "$source_dir/xls/pir_block_oracle/"
cd "$source_dir"
"${BAZEL:-bazel}" build "$@" //xls/pir_block_oracle:xls_block_oracle
install -m 755 bazel-bin/xls/pir_block_oracle/xls_block_oracle "$work/xls_block_oracle"
echo "$work/xls_block_oracle"
