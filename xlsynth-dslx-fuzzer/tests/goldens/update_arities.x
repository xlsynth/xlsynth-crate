// SPDX-License-Identifier: Apache-2.0


fn main(a0_v0: uN[3], a0_v1: uN[3], a1_v0: uN[3], a1_v1: uN[3], a2_v0: uN[3], a2_v1: uN[3], a3_v0: uN[3], a3_v1: uN[3], a4_v0: uN[3], a4_v1: uN[3], a5_v0: uN[3], a5_v1: uN[3]) -> (uN[3], uN[3], uN[3], uN[3], uN[3], uN[3],) {
    let a0_v2 = a0_v0 + a0_v1;
    let a0_v3 = { let matrix = [[a0_v0, a0_v1, a0_v0], [a0_v1, a0_v0, a0_v1]]; update(matrix, (), [[a0_v0 ^ a0_v2, a0_v1 + a0_v2, a0_v0 ^ a0_v2], [a0_v1 + a0_v2, a0_v0 ^ a0_v2, a0_v1 + a0_v2]])[u32:0][u32:0] };
    let a1_v2 = a1_v0 + a1_v1;
    let a1_v3 = { let matrix = [[a1_v0, a1_v1, a1_v0], [a1_v1, a1_v0, a1_v1]]; update(matrix, (), [[a1_v0 ^ a1_v2, a1_v1 + a1_v2, a1_v0 ^ a1_v2], [a1_v1 + a1_v2, a1_v0 ^ a1_v2, a1_v1 + a1_v2]])[u32:1][u32:0] };
    let a2_v2 = a2_v0 + a2_v1;
    let a2_v3 = { let matrix = [[a2_v0, a2_v1, a2_v0], [a2_v1, a2_v0, a2_v1]]; let row = ((a2_v0 as uN[3]) as u3) % u3:2; let coord = (row,); update(matrix, coord, [a2_v0 ^ a2_v2, a2_v1 + a2_v2, a2_v0 ^ a2_v2])[u32:0][u32:1] };
    let a3_v2 = a3_v0 + a3_v1;
    let a3_v3 = { let matrix = [[a3_v0, a3_v1, a3_v0], [a3_v1, a3_v0, a3_v1]]; let row = ((a3_v0 as uN[3]) as u3) % u3:2; let coord = (row,); update(matrix, coord, [a3_v0 ^ a3_v2, a3_v1 + a3_v2, a3_v0 ^ a3_v2])[u32:1][u32:0] };
    let a4_v2 = a4_v0 + a4_v1;
    let a4_v3 = { let cube = [[[a4_v0, a4_v1, a4_v0], [a4_v1, a4_v0, a4_v1]], [[a4_v1, a4_v0, a4_v1], [a4_v0, a4_v1, a4_v0]]]; let row = ((a4_v0 as uN[3]) as u3) % u3:2; let col = ((a4_v1 as uN[3]) as u3) % u3:3; let coord = ((a4_v2 as uN[3]) as u1, row, col as u32); update(cube, coord, a4_v2)[u32:0][u32:1][u32:2] };
    let a5_v2 = a5_v0 + a5_v1;
    let a5_v3 = { let cube = [[[a5_v0, a5_v1, a5_v0], [a5_v1, a5_v0, a5_v1]], [[a5_v1, a5_v0, a5_v1], [a5_v0, a5_v1, a5_v0]]]; let row = ((a5_v0 as uN[3]) as u3) % u3:2; let col = ((a5_v1 as uN[3]) as u3) % u3:3; let coord = ((a5_v2 as uN[3]) as u1, row, col as u32); update(cube, coord, a5_v2)[u32:1][u32:0][u32:1] };
    (a0_v3, a1_v3, a2_v3, a3_v3, a4_v3, a5_v3,)
}
