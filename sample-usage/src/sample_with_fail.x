// SPDX-License-Identifier: Apache-2.0

fn always_fail() -> () {
    fail!("this_always_fails", ())
}