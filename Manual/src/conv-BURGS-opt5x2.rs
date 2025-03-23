use image::{GrayImage, ImageBuffer, Luma};
use ndarray::{Array2, ArrayView2};
use rayon::prelude::*;

pub fn convolve(image: &GrayImage, kernel: &[&[f32]]) -> GrayImage {
    let (width, height) = image.dimensions();
    let img_array = Array2::from_shape_fn((height as usize, width as usize), |(y, x)| {
        image.get_pixel(x as u32, y as u32)[0] as f32
    });
    let kernel_height = kernel.len();
    let kernel_width = kernel[0].len();
    let kernel_array = Array2::from_shape_fn((kernel_height, kernel_width), |(y, x)| kernel[y][x]);
    let conv_height = height as usize - kernel_height + 1;
    let conv_width = width as usize - kernel_width + 1;
    let mut output = ImageBuffer::new(conv_width as u32, conv_height as u32);
    let block_size = 32;
    output
        .enumerate_pixels_mut()
        .par_bridge()
        .for_each(|(x, y, pixel)| {
            let block_x = (x as usize / block_size) * block_size;
            let block_y = (y as usize / block_size) * block_size;
            let x_end = (block_x + block_size).min(conv_width);
            let y_end = (block_y + block_size).min(conv_height);
            for by in block_y..y_end {
                for bx in block_x..x_end {
                    let window: ArrayView2<f32> =
                        img_array.slice(s![by..by + kernel_height, bx..bx + kernel_width]);
                    let val = window
                        .iter()
                        .zip(kernel_array.iter())
                        .map(|(a, k)| a * k)
                        .sum::<f32>()
                        .clamp(0.0, 255.0) as u8;
                    *output.get_pixel_mut(bx as u32, by as u32) = Luma([val]);
                }
            }
        });
    output
}