# Vim DSLX Runtime Files

SPDX-License-Identifier: Apache-2.0

This directory contains Vim runtime files for DSLX:

- `ftdetect/dslx.vim` assigns the `dslx` filetype to `*.x` and `*.dslx`.
- `ftplugin/dslx.vim` configures comment behavior for DSLX buffers.
- `syntax/dslx.vim` provides scanner-derived syntax highlighting.
- `testdata/feature_zoo.x` and `testdata/feature_zoo_imported.x` are small
  files for manual highlighting and typechecking checks.

## Installation

This directory is laid out as a Vim runtime directory. Vim will discover the
DSLX files when `extras/editors/vim/dslx` is on `runtimepath`.

### Vim 8 Native Packages

Create a package directory and symlink this runtime directory into it:

```bash
mkdir -p ~/.vim/pack/xlsynth/start
ln -s /path/to/xlsynth-crate/extras/editors/vim/dslx ~/.vim/pack/xlsynth/start/dslx
```

Restart Vim, or run `:helptags ALL` if you later add help files. Opening a
`*.x` or `*.dslx` file should set `filetype=dslx`.

### Existing `~/.vim` Runtime

If you do not use packages, copy or symlink the runtime subdirectories into
`~/.vim`:

```bash
mkdir -p ~/.vim/ftdetect ~/.vim/ftplugin ~/.vim/syntax
ln -s /path/to/xlsynth-crate/extras/editors/vim/dslx/ftdetect/dslx.vim ~/.vim/ftdetect/dslx.vim
ln -s /path/to/xlsynth-crate/extras/editors/vim/dslx/ftplugin/dslx.vim ~/.vim/ftplugin/dslx.vim
ln -s /path/to/xlsynth-crate/extras/editors/vim/dslx/syntax/dslx.vim ~/.vim/syntax/dslx.vim
```

### Neovim

The same files work in Neovim. Use Neovim's package directory:

```bash
mkdir -p ~/.local/share/nvim/site/pack/xlsynth/start
ln -s /path/to/xlsynth-crate/extras/editors/vim/dslx ~/.local/share/nvim/site/pack/xlsynth/start/dslx
```

### Manual Runtime Path

For quick local testing, add this directory to Vim's runtime path:

```vim
set runtimepath+=/path/to/xlsynth-crate/extras/editors/vim/dslx
```

The filetype detector assigns `filetype=dslx` for both `*.x` and `*.dslx`. This
is intentional: Vim otherwise treats `*.x` as `rpcgen` in a default setup. The
`.x` suffix is ambiguous in the wider editor ecosystem; adjust
`ftdetect/dslx.vim` locally if another `.x` language should win in your setup.

## Source Of Truth

The syntax rules are intentionally lexical and are derived from the
[XLS DSLX frontend](https://github.com/xlsynth/xlsynth/tree/main/xls/dslx) rather
than copied wholesale from the VS Code TextMate grammar:

- `xls/dslx/frontend/scanner_keywords.inc`: keywords and type keywords.
- `xls/dslx/frontend/ast_builtin_types.inc`: built-in type spellings.
- `xls/dslx/frontend/token.h`: token categories and punctuation.
- `xls/dslx/frontend/scanner.cc`: comments, literals, identifiers, and numbers.
- `xls/dslx/frontend/builtins_metadata.cc`: DSLX builtin functions/macros.
- `xls/dslx/highlight_main.cc`: existing high-level highlight categories.

The [DSLX VS Code extension](https://github.com/xlsynth/dslx-vscode) provides
useful editor configuration and broad syntax coverage, but its TextMate grammar
includes Rust-derived constructs such as block comments, raw strings, byte
strings, floats, octal literals, lifetimes, and `macro_rules!`. Those constructs
are not accepted by the current DSLX scanner, so this Vim syntax file does not
highlight them as DSLX syntax.

Some words are scanner keywords but are not part of the commonly supported DSLX
surface used by `xls/dslx/tests/*.x` and `xls/dslx/stdlib/*.x`. For example,
`trait` is behind `#![feature(traits)]` and normal parsing reports "Traits are
not yet supported"; `mut` is reserved by the scanner but is not used by the
test corpus. The Vim syntax highlights those as reserved words rather than
including them in the main keyword group.

## Validation

From this directory, typecheck the sample DSLX with an installed XLS build:

```bash
TYPECHECK_MAIN=/path/to/typecheck_main
DSLX_STDLIB_PATH=/path/to/xls/dslx/stdlib

"$TYPECHECK_MAIN" \
  --dslx_stdlib_path "$DSLX_STDLIB_PATH" \
  --dslx_path testdata \
  --output_path /tmp/dslx_feature_zoo_typeinfo.pb \
  testdata/feature_zoo.x
```

The sample includes proc/channel syntax, so `--output_path` is used to avoid the
textual type-info emitter path for channel types while still running parsing and
typechecking.

A minimal syntax-load check is:

```bash
vim -Nu NONE -n -es testdata/feature_zoo.x \
  -c 'set runtimepath^=.' \
  -c 'runtime ftdetect/dslx.vim' \
  -c 'set syntax=dslx' \
  -c 'syntax list dslxKeyword' \
  -c 'qa!'
```

For a stronger batch check, run the syntax/HTML smoke test:

```bash
vim -Nu NONE -n -es -S test_syntax_html.vim
```

That test probes Vim syntax groups directly and also runs Vim's built-in HTML
converter to verify that representative tokens are emitted with the expected
highlight classes. To keep the generated HTML for manual inspection:

```bash
vim -Nu NONE -n -es \
  -c 'let g:dslx_syntax_html_output="/tmp/dslx_feature_zoo.html"' \
  -c 'let g:dslx_syntax_keep_html=1' \
  -S test_syntax_html.vim
```
