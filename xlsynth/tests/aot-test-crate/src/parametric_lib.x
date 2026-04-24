// SPDX-License-Identifier: Apache-2.0

pub struct RemotePlain {
    id: u8,
}

pub struct RemoteBox<N: u32> {
    value: bits[N],
}

pub type RemoteBox8 = RemoteBox<u32:8>;

pub fn force_remote_box8(x: RemoteBox<u32:8>) -> RemoteBox<u32:8> {
    x
}
