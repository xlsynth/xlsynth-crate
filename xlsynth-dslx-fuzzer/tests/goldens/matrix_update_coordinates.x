// SPDX-License-Identifier: Apache-2.0


fn fuzz_coordinate(row: u3, col: u3) -> (u3, u32) { (row, col as u32) }

fn main(a0_v0: uN[3], a0_v1: uN[3], a1_v0: uN[3], a1_v1: uN[3], a2_v0: uN[3], a2_v1: uN[3], a3_v0: uN[3], a3_v1: uN[3]) -> (uN[3], uN[3], uN[3], uN[3],) {
    let a0_v2 = a0_v0 ^ a0_v1;
    let a0_v3 = { let matrix = [[a0_v0, a0_v1, a0_v0], [a0_v1, a0_v0, a0_v1]]; let row = ((a0_v0 as uN[3]) as u3) % u3:2; let col = ((a0_v1 as uN[3]) as u3) % u3:3; update(matrix, (row, col), a0_v2)[u32:1][u32:2] };
    let a1_v2 = a1_v0 ^ a1_v1;
    let a1_v3 = { let matrix = [[a1_v0, a1_v1, a1_v0], [a1_v1, a1_v0, a1_v1]]; let row = ((a1_v0 as uN[3]) as u3) % u3:2; let col = ((a1_v1 as uN[3]) as u3) % u3:3; let coord = (row, col); update(matrix, coord, a1_v2)[u32:1][u32:2] };
    let a2_v2 = a2_v0 ^ a2_v1;
    let a2_v3 = { let matrix = [[a2_v0, a2_v1, a2_v0], [a2_v1, a2_v0, a2_v1]]; let row = ((a2_v0 as uN[3]) as u3) % u3:2; let col = ((a2_v1 as uN[3]) as u3) % u3:3; let write = ((row, col), a2_v2); update(matrix, write.0, write.1)[u32:1][u32:2] };
    let a3_v2 = a3_v0 ^ a3_v1;
    let a3_v3 = { let matrix = [[a3_v0, a3_v1, a3_v0], [a3_v1, a3_v0, a3_v1]]; let row = ((a3_v0 as uN[3]) as u3) % u3:2; let col = ((a3_v1 as uN[3]) as u3) % u3:3; update(matrix, fuzz_coordinate(row, col), a3_v2)[u32:1][u32:2] };
    (a0_v3, a1_v3, a2_v3, a3_v3,)
}
