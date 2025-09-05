# syntax=docker/dockerfile:1.7-labs
# SPDX-License-Identifier: Apache-2.0
#
# Run via: `DOCKER_BUILDKIT=1 docker build -f codex/Dockerfile -t xlsynth-codex-offline .`

FROM ghcr.io/openai/codex-universal:latest

WORKDIR /workspace/xlsynth-crate

COPY . .

# Full toolchain and system deps, matching CI (nightly, clippy, protoc, llvm, etc.).
RUN bash codex/sample_codex_setup_script.sh

# Prepare artifacts and caches while the network is available.
RUN bash codex/sample_codex_maintenance_script.sh

# Provide persistent env for later steps by sourcing the env file the script wrote.
ENV CARGO_NET_OFFLINE=true

# Run tests and pre-commit offline; hooks and deps were fetched during maintenance.
RUN --network=none bash -lc 'set -a; . /etc/xlsynth/env; set +a; cargo nextest run --workspace'
RUN --network=none bash -lc 'set -a; . /etc/xlsynth/env; set +a; pre-commit run --all-files'
