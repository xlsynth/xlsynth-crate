use xlsynth::ir_value::IrBits;
use xlsynth::xlsynth_error::XlsynthError;

/// Generates an arbitrary IrBits value of the given width using the provided
/// random number generator. The width must be <= 64.
pub fn arbitrary_irbits<R: rand::Rng>(rng: &mut R, width: usize) -> Result<IrBits, XlsynthError> {
    assert!(width > 0, "width must be positive, got {}", width);
    assert!(
        width <= 64,
        "arbitrary_irbits only supports width <= 64 for now, got {}",
        width
    );
    let value = if width == 64 {
        rng.gen::<u64>()
    } else {
        rng.gen_range(0..(1u64 << width))
    };
    IrBits::make_ubits(width, value)
}
