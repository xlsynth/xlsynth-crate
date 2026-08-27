# Reference Docker image

This directory defines the Linux x86-64 contributor image used for local development
and CI. From the repository root, build it with Docker BuildKit (the default in
current Docker):

```shell
docker build --platform linux/amd64 --progress=plain \
  -f docker/Dockerfile -t xlsynth-dev-offline .
```

A successful build includes workspace `cargo check`, nextest tests, doctests, and
pre-commit, all with the container network disabled. Workspace checks and tests
enable `with-bitwuzla-system`, using the same pinned Bitwuzla binaries as CI.
The `container-build-and-test` CI job builds this same Dockerfile from a fresh checkout.
Preparation also compile-checks the solver-independent `fuzz_gate_fn_roundtrip`
target and the `xlsynth-vastly` fuzz targets used by pre-commit, with sanitizers disabled.

The image contains the source, resolved Cargo lockfiles, dependencies, matching XLS
DSO/DSLX standard library, and tool caches. Repeat its checks or open a shell without
mounting host caches or source:

```shell
docker run --rm --network=none xlsynth-dev-offline bash docker/check_offline.sh
docker run --rm -it --network=none xlsynth-dev-offline
```

The Dockerfile has two online preparation phases. `install_tools.sh` installs the
system packages, Rust toolchain, Cargo extensions, and solver DSOs in a reusable
layer. `prepare_workspace.sh` resolves the repository-specific XLS artifacts,
Cargo dependencies, and pre-commit environments, then writes
`~/.xlsynth_container_env.sh` for the offline check and interactive shells.

Inside the shell, use the same explicit solver feature:

```shell
cargo check --workspace --features with-bitwuzla-system --frozen
cargo test --workspace --features with-bitwuzla-system --frozen
```

Bitwuzla is an opt-in Cargo feature, but the default optimization and proof paths
require it. An unfeatured `cargo check` can succeed while an unfeatured workspace
`cargo test` fails on those paths; installing the DSO alone does not enable the
Rust feature. The separate CI solver matrix covers other solver configurations.

`CARGO_NET_OFFLINE=true` skips the crates.io version-comparison
tests, which check live publication state rather than build correctness. Other
workspace tests, including the local version-consistency checks, still run.

Compilation defaults to four jobs with incremental compilation and debug info
disabled to reduce memory and disk use. Override compilation/test concurrency with
`--build-arg BUILD_JOBS=2` on a smaller machine. On ARM hosts, the x86-64 image requires
Docker's emulation support; the release binaries used here are not Linux ARM builds.

## Preparation and reproducibility

The image pins Ubuntu by digest, Rust by nightly date, and cargo-nextest/cargo-fuzz
by version. Online preparation fetches all Cargo workspaces used by hooks before
any offline hook runs. XLS artifacts are resolved using
`scripts/get_required_libxls_dso_and_tools_release_tag.py`, stored under a
release/platform-specific directory, and supplied through `XLS_DSO_PATH`,
`DSLX_STDLIB_PATH`, and `LD_LIBRARY_PATH`. No globally installed XLS DSO is needed.

`.dockerignore` excludes host Git metadata, build outputs, Cargo lockfiles, and
downloaded artifacts. The image creates a local Git index solely for pre-commit,
so linked worktrees work too and host Git credentials are not copied.

The offline checks are hermetic with respect to networking, and `--frozen` preserves
the Cargo resolution captured during preparation. Rebuilding an image from the
same commit is **not yet bit-for-bit reproducible**: this repository does not commit
Cargo lockfiles, apt/Python package repositories can change, and the Slang `ci`
release is mutable. To reproduce a particular prepared environment, retain and
reuse that image by image ID/digest (or save it with `docker save`), rather than
rebuilding a mutable tag. The Cargo lockfiles remain in the image for inspection.
