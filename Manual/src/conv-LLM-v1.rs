// using rayon for parallel processing
use image::{GrayImage, ImageBuffer, Luma};
use rayon::prelude::*;

/// Applying convolution on a grayscale image using an optimized parallel implementation.
pub fn convolve(image: &GrayImage, kernel: &[&[f32]]) -> GrayImage {
    // computing kernel dimensions
    let (kh, kw) = (kernel.len(), kernel[0].len());

    // computing output image dimensions
    let (width, height) = image.dimensions();
    let output_width = width - kw as u32 + 1;
    let output_height = height - kh as u32 + 1;

    // allocating vector for output pixels
    let mut output_data = vec![0u8; (output_width * output_height) as usize];

    // parallelizing over rows in the output image
    (0..output_height).into_par_iter().for_each(|y| {
        // processing each row concurrently
        for x in 0..output_width {
            // initializing accumulator for convolution sum
            let mut acc = 0.0;
            for ky in 0..kh {
                // iterating over kernel rows
                for kx in 0..kw {
                    // fetching pixel value and applying corresponding kernel weight
                    let pixel_value = image.get_pixel(x + kx as u32, y + ky as u32)[0] as f32;
                    acc += pixel_value * kernel[ky][kx];
                }
            }
            // clamping the convolution result to the valid 8-bit range
            let clamped = acc.clamp(0.0, 255.0) as u8;
            // writing computed pixel value into the output vector
            let index = (y * output_width + x) as usize;
            output_data[index] = clamped;
        }
    });

    // constructing the output image from the computed pixel data
    ImageBuffer::from_vec(output_width, output_height, output_data)
        .expect("Failed constructing output image")
}
