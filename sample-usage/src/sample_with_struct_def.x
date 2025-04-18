// SPDX-License-Identifier: Apache-2.0

struct Point {
    x: u16,
    y: u32,
}

fn bump_coords(point: Point) -> Point {
    Point {
        x: point.x + u16:1,
        y: point.y + u32:1,
    }
}
