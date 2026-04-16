" SPDX-License-Identifier: Apache-2.0

augroup dslx_ftdetect
  autocmd!
  autocmd BufRead,BufNewFile *.x,*.dslx setfiletype dslx
augroup END
