#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "${script_dir}"

echo "Running Vim DSLX syntax/HTML smoke test..."
vim -Nu NONE -n -es -S test_syntax_html.vim

if [[ -z "${TYPECHECK_MAIN:-}" || -z "${DSLX_STDLIB_PATH:-}" ]]; then
  echo "Skipping DSLX typecheck fixture validation."
  echo "Set TYPECHECK_MAIN and DSLX_STDLIB_PATH to enable it."
  exit 0
fi

echo "Running DSLX typecheck fixture validation..."
"${TYPECHECK_MAIN}" \
  --dslx_stdlib_path "${DSLX_STDLIB_PATH}" \
  --dslx_path testdata \
  --output_path /tmp/dslx_feature_zoo_typeinfo.pb \
  testdata/feature_zoo.x

"${TYPECHECK_MAIN}" \
  --dslx_stdlib_path "${DSLX_STDLIB_PATH}" \
  --output_path /tmp/dslx_feature_zoo_imported_typeinfo.pb \
  testdata/feature_zoo_imported.x
