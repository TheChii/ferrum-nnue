//!Helper operation traits for NN inference.
//!
//! Contains SIMD-optimized implementations for AVX2-capable CPUs,
//! with automatic fallback to scalar operations.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub trait VecAdd<Rhs = Self> {
    fn vec_add(&mut self, other: &Self);
}

pub trait VecSub<Rhs = Self> {
    fn vec_sub(&mut self, other: &Self);
}

// ============================================================================
// SIMD-optimized VecAdd/VecSub for i16 arrays (used by accumulator)
// ============================================================================

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
impl<const SIZE: usize> VecAdd for [i16; SIZE] {
    #[inline]
    fn vec_add(&mut self, other: &Self) {
        let mut i = 0;
        
        // Process 16 i16 values at a time with AVX2
        unsafe {
            while i + 16 <= SIZE {
                let a = _mm256_loadu_si256(self.as_ptr().add(i) as *const __m256i);
                let b = _mm256_loadu_si256(other.as_ptr().add(i) as *const __m256i);
                let sum = _mm256_add_epi16(a, b);
                _mm256_storeu_si256(self.as_mut_ptr().add(i) as *mut __m256i, sum);
                i += 16;
            }
        }
        
        // Handle remainder
        while i < SIZE {
            self[i] = self[i].wrapping_add(other[i]);
            i += 1;
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
impl<const SIZE: usize> VecSub for [i16; SIZE] {
    #[inline]
    fn vec_sub(&mut self, other: &Self) {
        let mut i = 0;
        
        // Process 16 i16 values at a time with AVX2
        unsafe {
            while i + 16 <= SIZE {
                let a = _mm256_loadu_si256(self.as_ptr().add(i) as *const __m256i);
                let b = _mm256_loadu_si256(other.as_ptr().add(i) as *const __m256i);
                let diff = _mm256_sub_epi16(a, b);
                _mm256_storeu_si256(self.as_mut_ptr().add(i) as *mut __m256i, diff);
                i += 16;
            }
        }
        
        // Handle remainder
        while i < SIZE {
            self[i] = self[i].wrapping_sub(other[i]);
            i += 1;
        }
    }
}

// Fallback scalar implementations
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
macro_rules! vec_op_fallbacks {
    ($trait:ident, $fn:ident, $op:tt $(, $type:ty)*) => {
        $(impl<const SIZE: usize> $trait for [$type; SIZE] {
            #[inline]
            fn $fn(&mut self, other: &Self) {
                for (l, r) in self.iter_mut().zip(other) {
                    *l = l.$op(*r);
                }
            }
        })*
    };
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
vec_op_fallbacks!(VecAdd, vec_add, wrapping_add, i16);
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
vec_op_fallbacks!(VecSub, vec_sub, wrapping_sub, i16);

// Other types always use scalar (not hot path)
macro_rules! vec_op_scalar {
    ($trait:ident, $fn:ident, $op:tt $(, $type:ty)*) => {
        $(impl<const SIZE: usize> $trait for [$type; SIZE] {
            #[inline]
            fn $fn(&mut self, other: &Self) {
                for (l, r) in self.iter_mut().zip(other) {
                    *l = l.$op(*r);
                }
            }
        })*
    };
}

vec_op_scalar!(VecAdd, vec_add, wrapping_add, u8, i8, u16, u32, i32, u64, i64, u128, i128);
vec_op_scalar!(VecSub, vec_sub, wrapping_sub, u8, i8, u16, u32, i32, u64, i64, u128, i128);

// ============================================================================
// Dot product
// ============================================================================

pub trait Dot<Rhs=Self> {
    type Output;
    fn dot(&self, other: &Self) -> Self::Output;
}

// SIMD-optimized dot product for i8 arrays (used in dense layers)
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
impl<const SIZE: usize> Dot for [i8; SIZE] {
    type Output = i32;

    #[inline]
    fn dot(&self, other: &Self) -> Self::Output {
        let mut sum: i32 = 0;
        let mut i = 0;
        
        unsafe {
            let mut acc = _mm256_setzero_si256();
            
            // Process 32 i8 values at a time
            while i + 32 <= SIZE {
                let a = _mm256_loadu_si256(self.as_ptr().add(i) as *const __m256i);
                let b = _mm256_loadu_si256(other.as_ptr().add(i) as *const __m256i);
                
                // Multiply pairs and add horizontally: i8*i8 -> i16, then accumulate
                // _mm256_maddubs_epi16 treats first arg as unsigned, so we use madd approach
                let lo_a = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a, 0));
                let hi_a = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a, 1));
                let lo_b = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b, 0));
                let hi_b = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b, 1));
                
                let prod_lo = _mm256_madd_epi16(lo_a, lo_b);
                let prod_hi = _mm256_madd_epi16(hi_a, hi_b);
                
                acc = _mm256_add_epi32(acc, prod_lo);
                acc = _mm256_add_epi32(acc, prod_hi);
                
                i += 32;
            }
            
            // Horizontal sum of accumulator
            let acc_lo = _mm256_extracti128_si256(acc, 0);
            let acc_hi = _mm256_extracti128_si256(acc, 1);
            let sum128 = _mm_add_epi32(acc_lo, acc_hi);
            let sum64 = _mm_add_epi32(sum128, _mm_srli_si128(sum128, 8));
            let sum32 = _mm_add_epi32(sum64, _mm_srli_si128(sum64, 4));
            sum = _mm_cvtsi128_si32(sum32);
        }
        
        // Handle remainder
        while i < SIZE {
            sum += self[i] as i32 * other[i] as i32;
            i += 1;
        }
        
        sum
    }
}

// Fallback scalar dot for i8
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
impl<const SIZE: usize> Dot for [i8; SIZE] {
    type Output = i32;

    #[inline]
    fn dot(&self, other: &Self) -> Self::Output {
        self.iter().zip(other).map(|(&l, &r)| l as i32 * r as i32).sum()
    }
}

// Other dot products (not hot path, always scalar)
macro_rules! dot_product_scalar {
    ($($type:ty => $out:ty),*) => {
        $(impl<const SIZE: usize> Dot for [$type; SIZE] {
            type Output = $out;

            #[inline]
            fn dot(&self, other: &Self) -> Self::Output {
                self.iter().zip(other).map(|(&l, &r)| l as Self::Output * r as Self::Output).sum()
            }
        })*
    };
}

dot_product_scalar! {
    i16 => i32,
    i32 => i32,
    i64 => i64
}

// ============================================================================
// Clipped ReLU
// ============================================================================

pub trait ClippedRelu<O, const SIZE: usize> {
    fn clipped_relu(&self, scale: O, min: O, max: O, out: &mut [O; SIZE]);
}

// SIMD-optimized clipped ReLU for i16 -> i8 (accumulator to hidden layer input)
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
impl<const SIZE: usize> ClippedRelu<i8, SIZE> for [i16; SIZE] {
    #[inline]
    fn clipped_relu(&self, scale: i8, min: i8, max: i8, out: &mut [i8; SIZE]) {
        let mut i = 0;
        
        unsafe {
            let min_vec = _mm256_set1_epi16(min as i16);
            let max_vec = _mm256_set1_epi16(max as i16);
            
            // Process 32 i8 outputs (from 32 i16 inputs) at a time using two 256-bit loads
            while i + 32 <= SIZE {
                let v0 = _mm256_loadu_si256(self.as_ptr().add(i) as *const __m256i);
                let v1 = _mm256_loadu_si256(self.as_ptr().add(i + 16) as *const __m256i);
                
                // Shift and clamp
                let (clamped0, clamped1) = if scale == 0 {
                    // Fast path: no shift needed
                    (
                        _mm256_max_epi16(_mm256_min_epi16(v0, max_vec), min_vec),
                        _mm256_max_epi16(_mm256_min_epi16(v1, max_vec), min_vec)
                    )
                } else {
                    let shift_amt = _mm_cvtsi32_si128(scale as i32);
                    (
                        _mm256_max_epi16(_mm256_min_epi16(_mm256_sra_epi16(v0, shift_amt), max_vec), min_vec),
                        _mm256_max_epi16(_mm256_min_epi16(_mm256_sra_epi16(v1, shift_amt), max_vec), min_vec)
                    )
                };
                
                // Pack two 256-bit i16 vectors into one 256-bit i8 vector
                // _mm256_packs_epi16 packs with saturation: [a0..a15] [b0..b15] -> [a0..a7 b0..b7 a8..a15 b8..b15]
                // We need to permute to get the right order
                let packed = _mm256_packs_epi16(clamped0, clamped1);
                // Permute to fix lane ordering: [0,1,2,3] -> [0,2,1,3]
                let result = _mm256_permute4x64_epi64(packed, 0b11_01_10_00);
                
                _mm256_storeu_si256(out.as_mut_ptr().add(i) as *mut __m256i, result);
                i += 32;
            }
            
            // Handle 16-value chunks
            if i + 16 <= SIZE {
                let v = _mm256_loadu_si256(self.as_ptr().add(i) as *const __m256i);
                let clamped = if scale == 0 {
                    _mm256_max_epi16(_mm256_min_epi16(v, max_vec), min_vec)
                } else {
                    let shift_amt = _mm_cvtsi32_si128(scale as i32);
                    _mm256_max_epi16(_mm256_min_epi16(_mm256_sra_epi16(v, shift_amt), max_vec), min_vec)
                };
                
                let lo = _mm256_extracti128_si256(clamped, 0);
                let hi = _mm256_extracti128_si256(clamped, 1);
                let packed = _mm_packs_epi16(lo, hi);
                _mm_storeu_si128(out.as_mut_ptr().add(i) as *mut __m128i, packed);
                i += 16;
            }
        }
        
        // Handle remainder
        while i < SIZE {
            out[i] = ((self[i] >> scale as i16).clamp(min as i16, max as i16)) as i8;
            i += 1;
        }
    }
}

// SIMD-optimized clipped ReLU for i32 -> i8 (hidden layer output)
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
impl<const SIZE: usize> ClippedRelu<i8, SIZE> for [i32; SIZE] {
    #[inline]
    fn clipped_relu(&self, scale: i8, min: i8, max: i8, out: &mut [i8; SIZE]) {
        let mut i = 0;
        
        unsafe {
            let min_vec = _mm256_set1_epi32(min as i32);
            let max_vec = _mm256_set1_epi32(max as i32);
            let shift_amt = _mm_cvtsi32_si128(scale as i32);
            
            // Process 8 i32 values at a time
            while i + 8 <= SIZE {
                let v = _mm256_loadu_si256(self.as_ptr().add(i) as *const __m256i);
                
                // Shift right (arithmetic) - use runtime shift
                let shifted = _mm256_sra_epi32(v, shift_amt);
                
                // Clamp to [min, max]
                let clamped = _mm256_max_epi32(_mm256_min_epi32(shifted, max_vec), min_vec);
                
                // Pack i32 to i16 first, then i16 to i8
                let lo = _mm256_extracti128_si256(clamped, 0);
                let hi = _mm256_extracti128_si256(clamped, 1);
                let packed16 = _mm_packs_epi32(lo, hi);
                let packed8 = _mm_packs_epi16(packed16, _mm_setzero_si128());
                
                // Store 8 i8 values (only lower 64 bits valid)
                _mm_storel_epi64(out.as_mut_ptr().add(i) as *mut __m128i, packed8);
                i += 8;
            }
        }
        
        // Handle remainder
        while i < SIZE {
            out[i] = ((self[i] >> scale as i32).clamp(min as i32, max as i32)) as i8;
            i += 1;
        }
    }
}

// Fallback scalar implementations
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
macro_rules! clipped_relu_fallbacks {
    ($($type:ty => $out:ty),*) => {
        $(impl<const SIZE: usize> ClippedRelu<$out, SIZE> for [$type; SIZE] {
            #[inline]
            fn clipped_relu(&self, scale: $out, min: $out, max: $out, out: &mut [$out; SIZE]) {
                for (&v, o) in self.iter().zip(out) {
                    *o = (v >> scale as $type).clamp(min as $type, max as $type) as $out;
                }
            }
        })*
    };
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
clipped_relu_fallbacks! {
    i16 => i8,
    i32 => i8
}
