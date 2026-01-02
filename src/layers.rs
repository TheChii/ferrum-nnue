//!Module for layer types.
//!
//! Contains SIMD-optimized dense layer implementations.

use std::ops::AddAssign;

use crate::ops::*;

use bytemuck::Zeroable;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

///A dense layer.
#[derive(Debug, Clone, Zeroable)]
pub struct Dense<W: Zeroable, B: Zeroable, const INPUTS: usize, const OUTPUTS: usize> {
    pub weights: [[W; INPUTS]; OUTPUTS],
    pub biases: [B; OUTPUTS]
}

// SIMD-optimized activation for i8 inputs -> i32 outputs (the hot path)
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
impl<const INPUTS: usize, const OUTPUTS: usize> Dense<i8, i32, INPUTS, OUTPUTS> {
    #[inline]
    pub fn activate(&self, inputs: &[i8; INPUTS], outputs: &mut [i32; OUTPUTS]) {
        for (o, (bias, weights)) in outputs.iter_mut().zip(self.biases.iter().zip(&self.weights)) {
            *o = *bias + simd_dot_i8(inputs, weights);
        }
    }
}

/// SIMD dot product for i8 arrays - highly optimized
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
fn simd_dot_i8<const SIZE: usize>(a: &[i8; SIZE], b: &[i8; SIZE]) -> i32 {
    let mut i = 0;
    
    unsafe {
        let mut acc = _mm256_setzero_si256();
        
        // Process 32 i8 values at a time
        while i + 32 <= SIZE {
            let va = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
            let vb = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);
            
            // Split into low and high 128-bit halves, sign-extend to i16, multiply and add
            let lo_a = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(va, 0));
            let hi_a = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(va, 1));
            let lo_b = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vb, 0));
            let hi_b = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vb, 1));
            
            // madd: multiply pairs and add adjacent -> i32
            let prod_lo = _mm256_madd_epi16(lo_a, lo_b);
            let prod_hi = _mm256_madd_epi16(hi_a, hi_b);
            
            acc = _mm256_add_epi32(acc, prod_lo);
            acc = _mm256_add_epi32(acc, prod_hi);
            
            i += 32;
        }
        
        // Horizontal sum
        let acc_lo = _mm256_extracti128_si256(acc, 0);
        let acc_hi = _mm256_extracti128_si256(acc, 1);
        let sum128 = _mm_add_epi32(acc_lo, acc_hi);
        let sum64 = _mm_add_epi32(sum128, _mm_srli_si128(sum128, 8));
        let sum32 = _mm_add_epi32(sum64, _mm_srli_si128(sum64, 4));
        let mut sum = _mm_cvtsi128_si32(sum32);
        
        // Handle remainder
        while i < SIZE {
            sum += a[i] as i32 * b[i] as i32;
            i += 1;
        }
        
        sum
    }
}

// Fallback for non-AVX2
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
impl<
    W: Copy + Zeroable, B: Copy + Zeroable + AddAssign + From<<[W; INPUTS] as Dot>::Output>,
    const INPUTS: usize,
    const OUTPUTS: usize
> Dense<W, B, INPUTS, OUTPUTS> where [W; INPUTS]: Dot {
    #[inline]
    pub fn activate(&self, inputs: &[W; INPUTS], outputs: &mut [B; OUTPUTS]) {
        *outputs = self.biases;
        for (o, w) in outputs.iter_mut().zip(&self.weights) {
            *o += inputs.dot(w).into();
        }
    }
}

///A specialized [`Dense`] layer that operates on boolean inputs
///and can incrementally update the output accumulator.
#[derive(Debug, Clone, Zeroable)]
pub struct BitDense<WB: Zeroable, const INPUTS: usize, const OUTPUTS: usize> {
    pub weights: [[WB; OUTPUTS]; INPUTS],
    pub biases: [WB; OUTPUTS]
}

impl<
    WB: Zeroable + Clone,
    const INPUTS: usize,
    const OUTPUTS: usize
> BitDense<WB, INPUTS, OUTPUTS>
where
    [WB; OUTPUTS]: VecAdd + VecSub {
    ///Clear an accumulator to a default state.
    #[inline]
    pub fn empty(&self, outputs: &mut [WB; OUTPUTS]) {
        *outputs = self.biases.clone();
    }

    ///Add an input feature to an accumulator.
    #[inline]
    pub fn add(&self, index: usize, outputs: &mut [WB; OUTPUTS]) {
        outputs.vec_add(&self.weights[index]);
    }

    ///Remove an input feature from an accumulator.
    #[inline]
    pub fn sub(&self, index: usize, outputs: &mut [WB; OUTPUTS]) {
        outputs.vec_sub(&self.weights[index]);
    }

    ///Debug function for testing
    pub fn activate(&self, inputs: &[bool; INPUTS], outputs: &mut [WB; OUTPUTS]) {
        self.empty(outputs);
        for (i, &input) in inputs.iter().enumerate() {
            if input {
                self.add(i, outputs);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, Copy)]
    struct Rng(u128);
    
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_mul(0xDA942042E4DD58B5);
            (self.0 >> 64) as u64
        }
    }

    const RNG: Rng = Rng(0x576F77596F75466F756E645468697321);

    #[test]
    fn bitboard_dense_is_dense() {
        const INPUTS: usize = 64;
        const OUTPUTS: usize = 256;

        let mut rng = RNG;
        for _ in 0..100 {
            let mut dense = Dense {
                weights: [[0; INPUTS]; OUTPUTS],
                biases: [0; OUTPUTS]
            };
            let mut bit_dense = BitDense {
                weights: [[0; OUTPUTS]; INPUTS],
                biases: [0; OUTPUTS]
            };
            for output in 0..OUTPUTS {
                for input in 0..64 {
                    let weight = rng.next() as i8;
                    dense.weights[output][input] = weight;
                    bit_dense.weights[input][output] = weight;
                }
            }
            for (d, b) in dense.biases.iter_mut().zip(&mut bit_dense.biases) {
                let bias = rng.next() as i8;
                *d = bias as i32;
                *b = bias;
            }

            let bit_input = rng.next();
            let mut inputs = [0; 64];
            let mut bit_dense_inputs = [false; 64];
            for (i, (sq, sq2)) in inputs.iter_mut().zip(&mut bit_dense_inputs).enumerate() {
                *sq2 = ((bit_input >> i) & 1) != 0;
                *sq = *sq2 as i8;
            }
            
            let mut dense_output = [0; OUTPUTS];
            let mut bit_dense_output = [0; OUTPUTS];
            dense.activate(&inputs, &mut dense_output);
            bit_dense.activate(&bit_dense_inputs, &mut bit_dense_output);
            assert!(dense_output.iter().zip(&bit_dense_output).all(|(&d, &b)| d as i8 == b));
        }
    }
}