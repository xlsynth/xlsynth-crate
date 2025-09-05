# syntax=docker/dockerfile:1.7-labs
# SPDX-License-Identifier: Apache-2.0
#
# Run via: `DOCKER_BUILDKIT=1 docker build -f codex/Dockerfile -t xlsynth-codex-offline .`

FROM ghcr.io/openai/codex-universal:latest

ENV DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-lc"]

# Base packages needed early (git for pre-commit repo context, certs for TLS)
# Ensure tools needed by local scripts are present
RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo python3-pip ca-certificates && \
    update-ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    python3 -m pip install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir pre-commit

WORKDIR /workspace/xlsynth-crate

COPY . .

RUN chmod +x codex/sample_codex_setup_script.sh codex/sample_codex_maintenance_script.sh

# Full toolchain and system deps, matching CI (nightly, clippy, protoc, llvm, etc.).
RUN bash codex/sample_codex_setup_script.sh

# Prepare artifacts and caches while the network is available.
RUN bash codex/sample_codex_maintenance_script.sh

# Note: We assume the mounted working directory will already be a git repo at runtime.
# Tests and pre-commit are executed at runtime with the network disabled.

RUN cargo nextest run --workspace
RUN pre-commit run --all-files
