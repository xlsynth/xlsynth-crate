# syntax=docker/dockerfile:1
# SPDX-License-Identifier: Apache-2.0
#
# Run via: `docker build --platform linux/amd64 -f codex/Dockerfile -t xlsynth-codex-offline .`

# Ubuntu 24.04, Linux amd64. The downloaded XLS/protoc/Slang binaries are x86-64.
FROM ubuntu:24.04@sha256:1e0a86e57d247923571b75e0aaf48a1449cf8c543d51fb3e07a4a7d7bfa79316

ARG BUILD_JOBS=4
ENV RUSTUP_TOOLCHAIN=nightly-2026-08-27 \
    CARGO_BUILD_JOBS=${BUILD_JOBS} \
    CARGO_INCREMENTAL=0 \
    CARGO_PROFILE_DEV_DEBUG=0 \
    CARGO_PROFILE_TEST_DEBUG=0

WORKDIR /workspace/xlsynth-crate

# Cache tool installation separately from source changes.
COPY codex/sample_codex_setup_script.sh codex/
COPY scripts/ci_apt_retry.sh scripts/ci_install_rustup.py scripts/download_and_verify_shared_lib.py scripts/
RUN bash codex/sample_codex_setup_script.sh --tools-only

COPY . .
# Host Git metadata is deliberately excluded (including linked-worktree paths
# and credentials). Hooks need only an index of the source files in the image.
RUN git init -q && git add . && bash codex/sample_codex_maintenance_script.sh

ENV CARGO_NET_OFFLINE=true

# These commands cannot use the network, even from build scripts or tests.
RUN --network=none bash codex/check_offline.sh

CMD ["bash", "-l"]
