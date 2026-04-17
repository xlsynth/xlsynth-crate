#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage:
  ci_apt_retry.sh update
  ci_apt_retry.sh install [apt-get install args...]

Runs apt-get with CI-oriented retry settings. On later retries, Ubuntu archive
and security sources are rewritten to a fallback mirror.
EOF
}

configure_apt() {
  mkdir -p /etc/apt/apt.conf.d
  cat >/etc/apt/apt.conf.d/99ci-retries <<'EOF'
Acquire::ForceIPv4 "true";
Acquire::Retries "5";
Acquire::http::Timeout "30";
Acquire::https::Timeout "30";
Acquire::http::Pipeline-Depth "0";
EOF
}

rewrite_sources_file_to_fallback() {
  local path="$1"
  local fallback_mirror="$2"

  [ -f "${path}" ] || return 0
  cp -n "${path}" "${path}.ci-primary" 2>/dev/null || true
  sed -i \
    -e "s|http://archive.ubuntu.com/ubuntu|${fallback_mirror}|g" \
    -e "s|http://security.ubuntu.com/ubuntu|${fallback_mirror}|g" \
    -e "s|http://[a-z][a-z].archive.ubuntu.com/ubuntu|${fallback_mirror}|g" \
    "${path}"
}

use_fallback_mirror() {
  local fallback_mirror="${CI_APT_FALLBACK_MIRROR:-http://azure.archive.ubuntu.com/ubuntu}"

  echo "Switching Ubuntu apt sources to fallback mirror: ${fallback_mirror}" >&2
  rewrite_sources_file_to_fallback /etc/apt/sources.list "${fallback_mirror}"

  if [ -d /etc/apt/sources.list.d ]; then
    local source_file
    for source_file in /etc/apt/sources.list.d/*.list; do
      [ -e "${source_file}" ] || continue
      rewrite_sources_file_to_fallback "${source_file}" "${fallback_mirror}"
    done
  fi
}

refresh_sources_after_fallback() {
  local delay_seconds="${CI_APT_RETRY_DELAY_SECONDS:-10}"

  if apt-get update; then
    return 0
  fi
  echo "apt-get update failed after fallback mirror switch; retrying once" >&2
  sleep "${delay_seconds}"
  apt-get update
}

run_with_retries() {
  local attempts="${CI_APT_ATTEMPTS:-4}"
  local fallback_attempt="${CI_APT_FALLBACK_ATTEMPT:-3}"
  local delay_seconds="${CI_APT_RETRY_DELAY_SECONDS:-10}"
  local refresh_after_fallback="$1"
  shift

  local attempt=1
  local fallback_enabled=0
  while [ "${attempt}" -le "${attempts}" ]; do
    if [ "${attempt}" -eq "${fallback_attempt}" ] && [ "${fallback_enabled}" -eq 0 ]; then
      use_fallback_mirror
      fallback_enabled=1
      if [ "${refresh_after_fallback}" = "true" ]; then
        refresh_sources_after_fallback
      fi
    fi

    echo "apt attempt ${attempt}/${attempts}: $*" >&2
    if "$@"; then
      return 0
    fi

    if [ "${attempt}" -eq "${attempts}" ]; then
      echo "apt command failed after ${attempts} attempts: $*" >&2
      return 1
    fi

    attempt=$((attempt + 1))
    sleep "${delay_seconds}"
  done
}

main() {
  if [ "$#" -lt 1 ]; then
    usage
    exit 2
  fi

  export DEBIAN_FRONTEND="${DEBIAN_FRONTEND:-noninteractive}"
  configure_apt

  local subcommand="$1"
  shift
  case "${subcommand}" in
    update)
      if [ "$#" -ne 0 ]; then
        usage
        exit 2
      fi
      run_with_retries false apt-get update
      ;;
    install)
      if [ "$#" -eq 0 ]; then
        usage
        exit 2
      fi
      run_with_retries true apt-get install -y "$@"
      ;;
    *)
      usage
      exit 2
      ;;
  esac
}

main "$@"
