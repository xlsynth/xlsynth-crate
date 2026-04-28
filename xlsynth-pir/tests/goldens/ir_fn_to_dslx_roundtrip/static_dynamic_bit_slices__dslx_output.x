fn f(x: uN[8], i: uN[3]) -> uN[4] {
  let static_slice: uN[2] = x[2:4];
  let dynamic_slice: uN[2] = x[i+:uN[2]];
  let out_v: uN[4] = static_slice ++ dynamic_slice;
  out_v
}
