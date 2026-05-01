#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# This script is a CI regression test for the Cargo runtime-loader case that
# motivated the DSO staging logic in xlsynth-sys/build.rs.
#
# The important shape is:
#
#   downstream crate build.rs -> build-depends on xlsynth -> links libxls
#
# Cargo first links the downstream build-script executable successfully, then
# later starts that executable as a host binary. The regression was that libxls
# could be present in xlsynth-sys's OUT_DIR for link time, but absent from the
# dynamic loader path Cargo provides when it starts the downstream build script.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/xlsynth-build-script-dso.XXXXXX")"

# Keep the repro isolated from the workspace target dir. That makes the final
# target/debug/deps assertion specific to the throwaway downstream crate.
cleanup() {
  rm -rf "${tmp_dir}"
}
trap cleanup EXIT

mkdir -p "${tmp_dir}/src"

# Generate the smallest downstream package that still exercises the failure
# mode. xlsynth must be a build-dependency, not a normal dependency, because we
# need Cargo to run a build.rs executable that links through xlsynth-sys.
cat >"${tmp_dir}/Cargo.toml" <<EOF
[package]
name = "xlsynth-build-script-dso-check"
version = "0.1.0"
edition = "2021"
build = "build.rs"

[build-dependencies]
xlsynth = { path = "${repo_root}/xlsynth" }
EOF

# Touch an xlsynth API from build.rs so the generated build-script executable
# really links against libxls. A build script that merely depends on xlsynth but
# never references it can fail to reproduce the loader problem.
cat >"${tmp_dir}/build.rs" <<'EOF'
fn main() {
    let _ = xlsynth::xls_parse_typed_value("bits[8]:42").unwrap();
}
EOF

# Cargo still requires the package to have a crate target, but the target body
# is irrelevant; all interesting behavior happens while Cargo builds and runs
# the generated build script above.
cat >"${tmp_dir}/src/lib.rs" <<'EOF'
pub fn marker() {}
EOF

# Scrub loader and artifact override environment variables so the test covers
# the managed-download path. In particular:
#
# - LD_LIBRARY_PATH / DYLD_* must not mask a missing staged DSO.
# - XLS_DSO_PATH / DSLX_STDLIB_PATH / XLSYNTH_ARTIFACT_CONFIG must not switch
#   xlsynth-sys into the explicit pre-fetched artifact path.
# - DEV_XLS_DSO_WORKSPACE must not switch xlsynth-sys into the local XLS
#   workspace symlink path.
env \
  -u LD_LIBRARY_PATH \
  -u DYLD_LIBRARY_PATH \
  -u DYLD_FALLBACK_LIBRARY_PATH \
  -u XLS_DSO_PATH \
  -u DSLX_STDLIB_PATH \
  -u XLSYNTH_ARTIFACT_CONFIG \
  -u DEV_XLS_DSO_WORKSPACE \
  cargo check --manifest-path "${tmp_dir}/Cargo.toml"

# The fix stages managed libxls DSOs into the downstream Cargo profile's deps
# directory, which Cargo includes in the runtime loader environment for host
# binaries it runs. Check for the staged DSO explicitly so the test fails if
# Cargo happens to succeed for some unrelated reason.
#
# Use a shell glob instead of find: the Rocky CI image used by this repo is
# intentionally small and may not have find installed.
staged_dsos=("${tmp_dir}"/target/debug/deps/libxls-*)
if [[ ! -e "${staged_dsos[0]}" ]]; then
  echo "error: expected staged libxls DSO in ${tmp_dir}/target/debug/deps" >&2
  exit 1
fi
printf '%s\n' "${staged_dsos[0]}"
