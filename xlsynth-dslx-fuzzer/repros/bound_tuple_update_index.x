// SPDX-License-Identifier: Apache-2.0
fn main(row: u1, col: u1, word: u3) -> u3 {
    let zero_grid = [[u3:0, u3:0], [u3:0, u3:0]];
    let coord = (row, col);
    let updated = update(zero_grid, coord, word);
    updated[u1:0][u1:1]
}
