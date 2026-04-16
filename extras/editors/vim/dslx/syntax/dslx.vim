" SPDX-License-Identifier: Apache-2.0

if exists("b:current_syntax")
  finish
endif

syn case match

" Attributes such as #[test] and #![feature(use_syntax)].
syn keyword dslxAttributeName derive dslx_format_disable extern_verilog feature fuzz_test quickcheck sv_type test test_proc contained
syn region dslxAttribute start=+#!\=\[+ skip=+"\([^"\\]\|\\.\)*"\|`[^`]*`\|'\([^'\\]\|\\.\)'+ end=+\]+ contains=dslxAttributeName,dslxComment,dslxString,dslxBacktickString,dslxCharacter,dslxNumber,dslxBoolean,dslxKeyword,dslxType,dslxBuiltin,dslxMacro,dslxOperator

" String-like tokens accepted by the DSLX scanner. Backtick strings are valid
" in parser-supported attribute positions such as #[fuzz_test(domains=`...`)].
syn match dslxEscape "\\\(x[0-9A-Fa-f]\{2}\|u{[0-9A-Fa-f]\{1,6}}\|[nrt0\\'\"]\)" contained
syn match dslxInvalidEscape "\\." contained
syn region dslxString start=+"+ skip=+\\\\\|\\"+ end=+"+ contains=dslxEscape,dslxInvalidEscape
syn region dslxBacktickString start=+`+ end=+`+ contains=dslxEscape,dslxInvalidEscape
syn match dslxCharacter "'\([^'\\]\|\\\(x[0-9A-Fa-f]\{2}\|[nrt0\\'\"]\)\)'"

" Numeric spellings mirror Scanner::ScanNumber: decimal, binary, and hex.
syn match dslxNumber "\%(^\|[^A-Za-z0-9_']\)\zs-\?0x[0-9A-Fa-f_]\+\ze\%([^A-Za-z0-9_']\|$\)"
syn match dslxNumber "\%(^\|[^A-Za-z0-9_']\)\zs-\?0b[01_]\+\ze\%([^A-Za-z0-9_']\|$\)"
syn match dslxNumber "\%(^\|[^A-Za-z0-9_']\)\zs-\?\%(0\|[1-9][0-9]*\)\ze\%([^A-Za-z0-9_']\|$\)"

" Language keywords from xls/dslx/frontend/scanner_keywords.inc that are used
" in the parser/tests as normal DSLX surface syntax.
syn keyword dslxKeyword as const else enum fn for if impl import in let match out proc pub self spawn struct type use
syn keyword dslxBoolean true false

" Scanner-reserved or feature-gated words that are not normal DSLX today.
syn keyword dslxReserved mut trait

" Type keywords and sized primitive types from scanner_keywords.inc.
syn keyword dslxType bits token uN sN xN bool chan Self
syn match dslxType "\<[us]\%([1-9]\|[1-5][0-9]\|6[0-4]\)\>"

" Builtins from xls/dslx/frontend/builtins_metadata.cc. The scanner treats
" token as a type keyword, so it is intentionally not repeated here.
syn keyword dslxBuiltin assert_eq assert_lt bit_slice_update ceillog2 clz configured_value_or ctz map decode encode one_hot one_hot_sel priority_sel rev umulp smulp array_rev array_size bit_count element_count and_reduce or_reduce xor_reduce signex array_slice update enumerate widening_cast checked_cast range zip labeled_read labeled_write send send_if recv recv_if recv_non_blocking recv_if_non_blocking join read write
syn match dslxMacro "\%(^\|[^A-Za-z0-9_']\)\zs\%(cover\|fail\|assert\|trace\|all_ones\|zero\|trace_fmt\|assert_fmt\|vtrace_fmt\|gate\|const_assert\|unroll_for\)!\ze\%([^A-Za-z0-9_']\|$\)"

" Lightweight semantic cues. These are regex-based and deliberately do not try
" to parse nested DSLX.
syn match dslxTypeName "\<[A-Z][A-Za-z0-9_']*\>"
syn match dslxFunctionDef "\<fn\s\+\zs[A-Za-z_][A-Za-z0-9_']*"
syn match dslxProcDef "\<proc\s\+\zs[A-Za-z_][A-Za-z0-9_']*"
syn match dslxProcMemberDef "^\s*\zs\(config\|init\|next\)\ze\s*[({]"
syn match dslxConstDef "\<const\s\+\zs[A-Z][A-Z0-9_']*"
syn match dslxTypeDef "\<\(struct\|enum\|type\)\s\+\zs[A-Za-z_][A-Za-z0-9_']*"
syn match dslxLetBinding "\<let\s\+\zs[A-Za-z_][A-Za-z0-9_']*"
syn match dslxEnumVariant "::\zs[A-Za-z_][A-Za-z0-9_']*"
syn match dslxFieldAccess "\.\zs[A-Za-z_][A-Za-z0-9_']*"
syn match dslxFunctionCall "\<[A-Za-z_][A-Za-z0-9_']*\ze\s*<[^>\n]\+>\s*("
syn match dslxFunctionCall "\<[A-Za-z_][A-Za-z0-9_']*\ze\s*("
syn match dslxNamespace "\<[A-Za-z_][A-Za-z0-9_']*\ze::"
syn match dslxConstant "\<[A-Z][A-Z0-9_]*\>"

" Token punctuation from xls/dslx/frontend/token.h.
syn match dslxOperator "->\|=>\|::\|++\|+:\|<<\|>>\|<=\|>=\|==\|!="
syn match dslxOperator "&&\|||\|\.\.=\|\.\.\.\|\.\."
syn match dslxOperator "[+\-*/%&|^!~<>=:.]"
syn match dslxDelimiter "[,;()[\]{}]"

" DSLX comments are line comments only in the frontend scanner. Define comments
" after normal lexical groups so comment text is not re-highlighted as code.
syn keyword dslxTodo TODO FIXME XXX NOTE contained
syn match dslxComment "//.*$" contains=dslxTodo

hi def link dslxTodo Todo
hi def link dslxComment Comment
hi def link dslxAttribute PreProc
hi def link dslxAttributeName PreProc
hi def link dslxEscape SpecialChar
hi def link dslxInvalidEscape Error
hi def link dslxString String
hi def link dslxBacktickString String
hi def link dslxCharacter Character
hi def link dslxNumber Number
hi def link dslxKeyword Keyword
hi def link dslxBoolean Boolean
hi def link dslxReserved Statement
hi def link dslxType Type
hi def link dslxBuiltin Function
hi def link dslxMacro Macro
hi def link dslxFunctionDef Function
hi def link dslxProcDef Function
hi def link dslxProcMemberDef Function
hi def link dslxConstDef Constant
hi def link dslxTypeDef Type
hi def link dslxTypeName Type
hi def link dslxLetBinding Identifier
hi def link dslxEnumVariant Constant
hi def link dslxFieldAccess Identifier
hi def link dslxFunctionCall Function
hi def link dslxNamespace Identifier
hi def link dslxConstant Constant
hi def link dslxOperator Operator
hi def link dslxDelimiter Delimiter

syn sync minlines=50

let b:current_syntax = "dslx"
