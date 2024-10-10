// SPDX-License-Identifier: Apache-2.0

pub enum MyEnum : u8 {
    A = u8:2,
    B = u8:42,
    C = u8:255,
}

fn my_enum_to_u8(my_enum: MyEnum) -> u8 {
    my_enum as u8
}