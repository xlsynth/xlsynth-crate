These are support materials for getting a Codex (web) environment working with the code base.

The setup script should be "cacheable" actions whereas the maintenance script is re-run for each
session.

The Dockerfile in this directory is to enable testing of a similar environment/setup to make sure
script modifications seem sound before trying in the Codex web environment.

Notes:

- The repo expects a specific XLS release tag (as defined by `xlsynth-sys/build.rs`). The
  maintenance script uses `scripts/get_required_xls_release_tag.py` to fetch the matching DSO and
  DSLX stdlib, and sets `XLS_DSO_PATH` and `DSLX_STDLIB_PATH` so `cargo test` and pre-commit hooks
  do not attempt network downloads.
