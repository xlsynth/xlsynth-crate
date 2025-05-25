# AGENTS.md

All pull requests must be clean with respect to `pre-commit`.

To verify locally, run:

```bash
pre-commit run --all-files
```

PRs that fail this check will not be accepted.

## Agent Guidance: xlsynth-g8r and Fuzz Targets

If you are modifying code in the `xlsynth-g8r` crate, you **must** ensure that all related fuzz targets (such as those in `xlsynth-g8r/fuzz/fuzz_targets`) still build. CI will fail if any fuzz target does not build. Always check the build status of these fuzz targets after making changes to `xlsynth-g8r`.
