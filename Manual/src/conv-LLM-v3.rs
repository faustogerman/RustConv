use image::{GrayImage, ImageBuffer, Luma};
use rayon::prelude::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Applying optimized convolution on a grayscale image using SIMD vectorization,
/// cache blocking (processing blocks of output pixels), loop unrolling, and parallel row processing.
pub fn convolve(image: &GrayImage, kernel: &[&[f32]]) -> GrayImage {
    // computing kernel dimensions
    let (kh, kw) = (kernel.len(), kernel[0].len());
    // computing input and output image dimensions
    let (width, height) = image.dimensions();
    let output_width = width - kw as u32 + 1;
    let output_height = height - kh as u32 + 1;

    // extracting raw pixel data for improved cache locality
    let input_buffer = image.as_raw();
    // allocating vector for output pixels
    let mut output_data = vec![0u8; (output_width * output_height) as usize];

    if kh == 3 && kw == 3 {
        // if target supports SSE4.1, using SIMD-accelerated 3×3 convolution
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("sse4.1") {
                unsafe {
                    simd_convolve_3x3(
                        input_buffer,
                        kernel,
                        width as usize,
                        output_width as usize,
                        output_height as usize,
                        &mut output_data,
                    );
                    return ImageBuffer::from_vec(output_width, output_height, output_data)
                        .expect("Failed constructing output image");
                }
            }
        }
        // falling back to scalar unrolled 3×3 convolution if SIMD is not available
        scalar_unrolled_convolve_3x3(
            input_buffer,
            kernel,
            width as usize,
            output_width as usize,
            output_height as usize,
            &mut output_data,
        );
    } else {
        // using generic convolution for kernels other than 3×3, with raw buffer indexing
        output_data.par_chunks_mut(output_width as usize)
            .enumerate()
            .for_each(|(y, row)| {
                for x in 0..output_width as usize {
                    let mut acc = 0.0;
                    for ky in 0..kh {
                        let row_index = ((y + ky) * width as usize) as usize;
                        for kx in 0..kw {
                            let pixel = input_buffer[row_index + x + kx] as f32;
                            acc += pixel * kernel[ky][kx];
                        }
                    }
                    row[x] = acc.clamp(0.0, 255.0) as u8;
                }
            });
    }

    ImageBuffer::from_vec(output_width, output_height, output_data)
        .expect("Failed constructing output image")
}

/// Falling back to scalar loop unrolling for 3×3 kernels.
/// Processing each output pixel by unrolling the 3×3 kernel multiplication explicitly.
fn scalar_unrolled_convolve_3x3(
    input_buffer: &Vec<u8>,
    kernel: &[&[f32]],
    width: usize,
    output_width: usize,
    output_height: usize,
    output_data: &mut Vec<u8>,
) {
    // extracting kernel coefficients for unrolling
    let k00 = kernel[0][0];
    let k01 = kernel[0][1];
    let k02 = kernel[0][2];
    let k10 = kernel[1][0];
    let k11 = kernel[1][1];
    let k12 = kernel[1][2];
    let k20 = kernel[2][0];
    let k21 = kernel[2][1];
    let k22 = kernel[2][2];

    output_data.par_chunks_mut(output_width)
        .enumerate()
        .for_each(|(y, row)| {
            for x in 0..output_width {
                let base_index = y * width + x;
                let acc = (input_buffer[base_index] as f32 * k00)
                    + (input_buffer[base_index + 1] as f32 * k01)
                    + (input_buffer[base_index + 2] as f32 * k02)
                    + (input_buffer[base_index + width] as f32 * k10)
                    + (input_buffer[base_index + width + 1] as f32 * k11)
                    + (input_buffer[base_index + width + 2] as f32 * k12)
                    + (input_buffer[base_index + 2 * width] as f32 * k20)
                    + (input_buffer[base_index + 2 * width + 1] as f32 * k21)
                    + (input_buffer[base_index + 2 * width + 2] as f32 * k22);
                row[x] = acc.clamp(0.0, 255.0) as u8;
            }
        });
}

/// SIMD-accelerated convolution for 3×3 kernels processing 4 output pixels concurrently using SSE4.1 intrinsics.
/// Using cache blocking by processing a block (4 pixels) per iteration.
#[cfg(target_arch = "x86_64")]
unsafe fn simd_convolve_3x3(
    input_buffer: &Vec<u8>,
    kernel: &[&[f32]],
    width: usize,
    output_width: usize,
    output_height: usize,
    output_data: &mut Vec<u8>,
) {
    // broadcasting kernel coefficients for each kernel row
    let k0 = _mm_set_ps1(kernel[0][0]);
    let k1 = _mm_set_ps1(kernel[0][1]);
    let k2 = _mm_set_ps1(kernel[0][2]);
    let k3 = _mm_set_ps1(kernel[1][0]);
    let k4 = _mm_set_ps1(kernel[1][1]);
    let k5 = _mm_set_ps1(kernel[1][2]);
    let k6 = _mm_set_ps1(kernel[2][0]);
    let k7 = _mm_set_ps1(kernel[2][1]);
    let k8 = _mm_set_ps1(kernel[2][2]);
    let min_val = _mm_set_ps1(0.0);
    let max_val = _mm_set_ps1(255.0);

    // processing output rows in parallel using Rayon
    output_data
        .par_chunks_mut(output_width)
        .enumerate()
        .for_each(|(y, row)| {
            let mut x = 0;
            while x < output_width {
                // initializing sum vector for 4 output pixels
                let mut sum = _mm_setzero_ps();
                // processing each kernel row (i = 0,1,2)
                for i in 0..3 {
                    // computing pointer for input row (y + i) and column offset x
                    let row_ptr = input_buffer.as_ptr().add((y + i) * width + x);
                    // loading 4 u8 values from row_ptr for offset 0
                    let data0 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(*(row_ptr as *const i32)));
                    let v0 = _mm_cvtepi32_ps(data0);
                    // loading 4 u8 values from row_ptr + 1 for offset 1
                    let data1 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(*(row_ptr.add(1) as *const i32)));
                    let v1 = _mm_cvtepi32_ps(data1);
                    // loading 4 u8 values from row_ptr + 2 for offset 2
                    let data2 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(*(row_ptr.add(2) as *const i32)));
                    let v2 = _mm_cvtepi32_ps(data2);

                    // selecting appropriate kernel coefficients based on row offset
                    let (c0, c1, c2) = match i {
                        0 => (k0, k1, k2),
                        1 => (k3, k4, k5),
                        2 => (k6, k7, k8),
                        _ => (_mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps()),
                    };

                    // accumulating the convolution sum for this kernel row
                    let part = _mm_add_ps(
                        _mm_add_ps(_mm_mul_ps(v0, c0), _mm_mul_ps(v1, c1)),
                        _mm_mul_ps(v2, c2),
                    );
                    sum = _mm_add_ps(sum, part);
                }
                // clamping sum values to the valid range [0, 255]
                let sum_clamped = _mm_min_ps(_mm_max_ps(sum, min_val), max_val);
                // converting the 4 float values to 32-bit integers
                let sum_i32 = _mm_cvtps_epi32(sum_clamped);
                // storing the result into a temporary array
                let mut temp = [0i32; 4];
                _mm_storeu_si128(temp.as_mut_ptr() as *mut __m128i, sum_i32);
                // writing computed pixel values to the output row
                for j in 0..4 {
                    if x + j < output_width {
                        row[x + j] = temp[j] as u8;
                    }
                }
                x += 4;
            }
        });
}
