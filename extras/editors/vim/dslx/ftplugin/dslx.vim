" SPDX-License-Identifier: Apache-2.0

if exists("b:did_ftplugin")
  finish
endif
let b:did_ftplugin = 1

let b:undo_ftplugin = "setlocal commentstring< comments< formatoptions< suffixesadd<"

setlocal commentstring=//\ %s
setlocal comments=://
setlocal suffixesadd=.x,.dslx
setlocal formatoptions-=t
setlocal formatoptions+=croql
