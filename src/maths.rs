/// Fast Exponential function.
#[inline]
pub fn fast_exp(p: f32) -> f32 {
    pow2(std::f32::consts::LOG2_E * p)
}

#[inline]
fn pow2(p: f32) -> f32 {
    let clipp = if p < -126.0 { -126.0 } else { p };
    let v = ((1 << 23) as f32 * (clipp + 126.942696)) as u32;
    f32::from_bits(v)
}
