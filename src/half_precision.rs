//! `F16`/`bfloat16` <-> `f32` bit-manipulation conversions, ported from
//! `magnetar-runtime`'s already-verified implementation (`f16_to_f32`/
//! `bf16_to_f32`/`f32_to_f16`/`f32_to_bf16` in `model_loading.rs`, landed by
//! `add-native-cuda-half-precision-compute`'s Phase 1) -- this crate does
//! not depend on `magnetar-runtime` internals for it, matching the
//! established "ported, not shared" convention externalized modules use for
//! logic another independent externalized module already has its own
//! verified copy of (e.g. `loaders/gguf`'s own `dequantize.rs`, ported from
//! this same original). Used by [`crate::kernels::CudaKernels`]'s
//! `upload_half`/`download_half` to convert host `f32` values to/from
//! on-device half-precision bytes at the upload/download boundary.

/// Converts one IEEE 754 binary16 value to `f32`, exactly.
pub(crate) fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15) << 31;
    let exponent = (bits >> 10) & 0x1F;
    let mantissa = u32::from(bits & 0x3FF);
    let magnitude_bits = if exponent == 0 {
        if mantissa == 0 {
            0
        } else {
            let mut mantissa = mantissa;
            let mut shift = 0u32;
            while mantissa & 0x400 == 0 {
                mantissa <<= 1;
                shift += 1;
            }
            mantissa &= 0x3FF;
            let f32_exponent = 127 - 15 - shift + 1;
            (f32_exponent << 23) | (mantissa << 13)
        }
    } else if exponent == 0x1F {
        (0xFFu32 << 23) | (mantissa << 13)
    } else {
        let f32_exponent = (i32::from(exponent) - 15 + 127) as u32;
        (f32_exponent << 23) | (mantissa << 13)
    };
    f32::from_bits(sign | magnitude_bits)
}

/// Converts one `bfloat16` value to `f32`, exactly.
pub(crate) fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// Converts one `f32` value to IEEE 754 binary16, rounding to nearest with
/// ties to even.
pub(crate) fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let mantissa_f32 = bits & 0x007F_FFFF;
    let exp_f32 = ((bits >> 23) & 0xFF) as i32;

    if exp_f32 == 0xFF {
        if mantissa_f32 == 0 {
            return sign | 0x7C00;
        }
        let payload = (mantissa_f32 >> 13) as u16;
        let payload = if payload == 0 { 1 } else { payload };
        return sign | 0x7C00 | payload;
    }

    let unbiased = exp_f32 - 127;

    if unbiased > 15 {
        return sign | 0x7C00;
    }

    if unbiased < -14 {
        let significand = if exp_f32 == 0 {
            mantissa_f32
        } else {
            mantissa_f32 | 0x0080_0000
        };
        if significand == 0 {
            return sign;
        }
        let shift = (-(unbiased + 1)) as u32;
        if shift >= 32 {
            return sign;
        }
        let half = 1u32 << (shift - 1);
        let mask = (1u32 << shift) - 1;
        let truncated = significand >> shift;
        let remainder = significand & mask;
        let mut result = truncated;
        if remainder > half || (remainder == half && (result & 1) == 1) {
            result += 1;
        }
        return sign | (result as u16);
    }

    let half = 1u32 << 12;
    let mask = (1u32 << 13) - 1;
    let truncated = mantissa_f32 >> 13;
    let remainder = mantissa_f32 & mask;
    let mut mantissa10 = truncated;
    if remainder > half || (remainder == half && (mantissa10 & 1) == 1) {
        mantissa10 += 1;
    }
    let exp16 = (unbiased + 15) as u32;
    if mantissa10 == 0x400 {
        let new_exp = exp16 + 1;
        if new_exp >= 0x1F {
            return sign | 0x7C00;
        }
        return sign | ((new_exp as u16) << 10);
    }
    sign | ((exp16 as u16) << 10) | (mantissa10 as u16)
}

/// Converts one `f32` value to `bfloat16`, rounding to nearest with ties to
/// even.
pub(crate) fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    if value.is_nan() {
        let sign = ((bits >> 16) & 0x8000) as u16;
        let mantissa7 = ((bits >> 16) as u16) & 0x7F;
        let mantissa7 = if mantissa7 == 0 { 0x40 } else { mantissa7 };
        return sign | 0x7F80 | mantissa7;
    }
    let rounding_bias = 0x0000_7FFFu32 + ((bits >> 16) & 1);
    let rounded = bits.wrapping_add(rounding_bias);
    (rounded >> 16) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Same exhaustive-round-trip proof as `magnetar-runtime`'s original
    /// (`add-native-cuda-half-precision-compute` Phase 1): every one of the
    /// 65,536 possible `u16` bit patterns is, by construction, an exactly
    /// representable `f16` value, so decoding and re-encoding it must
    /// reproduce the original bit pattern exactly (NaN payload excepted).
    #[test]
    fn f32_to_f16_round_trips_every_possible_f16_bit_pattern_exactly() {
        for bits in 0u32..=0xFFFF {
            let bits = bits as u16;
            let decoded = f16_to_f32(bits);
            let reencoded = f32_to_f16(decoded);
            if decoded.is_nan() {
                assert!(f16_to_f32(reencoded).is_nan());
            } else {
                assert_eq!(reencoded, bits);
            }
        }
    }

    #[test]
    fn f32_to_bf16_round_trips_every_possible_bf16_bit_pattern_exactly() {
        for bits in 0u32..=0xFFFF {
            let bits = bits as u16;
            let decoded = bf16_to_f32(bits);
            let reencoded = f32_to_bf16(decoded);
            if decoded.is_nan() {
                assert!(bf16_to_f32(reencoded).is_nan());
            } else {
                assert_eq!(reencoded, bits);
            }
        }
    }
}
