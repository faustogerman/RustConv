use image::{GrayImage, ImageBuffer, Luma};

use rayon::prelude::*;

fn par_convolve_pixel(
    image: &GrayImage,
    kernel: &[&[f32]],
    x: u32,
    y: u32,
    kh: usize,
    kw: usize,
) -> f32 {
    // Compute the sum of pixel_value * weight in parallel
    (0..kh)
        .into_par_iter() // parallel iterator over ky
        .map(|ky| {
            (0..kw) // you could also use .into_par_iter() here if kw is large enough
                .map(|kx| {
                    let pixel_value = image.get_pixel(x + kx as u32, y + ky as u32)[0] as f32;
                    let weight = kernel[ky][kx];
                    pixel_value * weight
                })
                .sum::<f32>()
        })
        .sum()
}

fn convolve_pixel(
    image: &GrayImage,
    kernel: &[&[f32]],
    x: &u32,
    y: &u32,
    kh: usize,
    kw: usize,
) -> f32 {
    let mut acc = 0.0;

    for ky in 0..kh {
        for kx in 0..kw {
            let pixel_value = image.get_pixel(x + kx as u32, y + ky as u32)[0] as f32;
            let weight = kernel[ky][kx];
            acc += pixel_value * weight;
        }
    }

    acc
}

/// Applies a convolution operation on a grayscale image using a kernel of any numeric type and size.
pub fn convolve(image: &GrayImage, kernel: &[&[f32]]) -> GrayImage {
    let (kh, kw) = (kernel.len(), kernel[0].len());

    let (width, height) = image.dimensions();
    let output_width = width - kw as u32 + 1;
    let output_height = height - kh as u32 + 1;

    let out_im: Vec<u8> = (0..output_height)
        .into_par_iter()
        .flat_map(|y| {
            (0..output_width)
                .into_par_iter()
                .map(|x| convolve_pixel(image, kernel, &x, &y, kh, kw).clamp(0.0, 255.0) as u8)
                .collect::<Vec<u8>>()
        })
        .collect();

    // Create image buffer from the flattened vector
    ImageBuffer::from_vec(output_width, output_height, out_im)
        .expect("Failed to create image buffer")
}

/// Applies a convolution operation on a grayscale image using a kernel of any numeric type and size.
pub fn convolve_v1(image: &GrayImage, kernel: &[&[f32]]) -> GrayImage {
    let (kh, kw) = (kernel.len(), kernel[0].len());

    let (width, height) = image.dimensions();
    let output_width = width - kw as u32 + 1;
    let output_height = height - kh as u32 + 1;

    let mut output = ImageBuffer::new(output_width, output_height);

    for y in 0..output_height {
        for x in 0..output_width {
            let mut acc = 0.0;

            for ky in 0..kh {
                for kx in 0..kw {
                    let pixel_value = image.get_pixel(x + kx as u32, y + ky as u32)[0] as f32;
                    let weight = kernel[ky][kx];
                    acc += pixel_value * weight;
                }
            }

            output.put_pixel(x, y, Luma([acc.clamp(0.0, 255.0) as u8]));
        }
    }

    output
}
