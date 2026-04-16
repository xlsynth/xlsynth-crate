" SPDX-License-Identifier: Apache-2.0

set nomore

let s:root = expand('<sfile>:p:h')
execute 'set runtimepath^=' . fnameescape(s:root)

filetype on
syntax on

function! s:Fail(message) abort
  echoerr a:message
  cquit
endfunction

function! s:AssertSyntaxAt(needle, offset, want) abort
  call cursor(1, 1)
  let lnum = search('\V' . escape(a:needle, '\'), 'W')
  if lnum == 0
    call s:Fail('could not find syntax probe: ' . a:needle)
  endif

  let col = col('.') + a:offset
  let got = synIDattr(synID(lnum, col, 1), 'name')
  if got !=# a:want
    call s:Fail(
          \ printf('syntax group for %s at %d:%d: got %s, want %s',
          \        string(a:needle), lnum, col, got, a:want))
  endif
endfunction

execute 'edit ' . fnameescape(s:root . '/testdata/feature_zoo.x')

if &filetype !=# 'dslx'
  call s:Fail('filetype=' . &filetype . ', want dslx')
endif

if &syntax !=# 'dslx'
  call s:Fail('syntax=' . &syntax . ', want dslx')
endif

syntax sync fromstart

call s:AssertSyntaxAt('// Keep this', 0, 'dslxComment')
call s:AssertSyntaxAt('// Keep this', 3, 'dslxComment')
call s:AssertSyntaxAt('impl Packet', 0, 'dslxKeyword')
call s:AssertSyntaxAt('struct Packet', 0, 'dslxKeyword')
call s:AssertSyntaxAt('trace_fmt!', 0, 'dslxMacro')
call s:AssertSyntaxAt('zero!<uN', 0, 'dslxMacro')

let g:html_no_progress = 1
let g:html_use_css = 1
let g:html_number_lines = 0

let s:html_output = get(g:, 'dslx_syntax_html_output', tempname() . '.html')

silent runtime syntax/2html.vim
silent execute 'write! ' . fnameescape(s:html_output)

let s:html = join(readfile(s:html_output), "\n")
let s:comment_html = '<span class="Comment">// Keep this file broad enough to exercise highlighting.</span>'

if stridx(s:html, s:comment_html) < 0
  call s:Fail('HTML output did not contain the expected single Comment span')
endif

if s:html !~# '<span class="Statement">impl</span>'
  call s:Fail('HTML output did not classify impl as Statement')
endif

if s:html !~# '<span class="PreProc">trace_fmt!</span>'
  call s:Fail('HTML output did not classify trace_fmt! as PreProc')
endif

if !get(g:, 'dslx_syntax_keep_html', 0)
  call delete(s:html_output)
endif

qa!
