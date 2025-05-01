use image::{GrayImage, ImageBuffer, Luma};
use ndarray::{Array2, s};
use ndarray::Zip;
use ndarray::parallel::prelude::*;
use rayon::prelude::*;

pub fn convolve(image: &GrayImage, kernel: &[&[f32]]) -> GrayImage {
    let (kh, kw) = (kernel.len(), kernel[0].len());
    let (width, height) = image.dimensions();
    let output_width = width - kw as u32 + 1;
    let output_height = height - kh as u32 + 1;
    let input_array: Array2<f32> = Array2::from_shape_fn((height as usize, width as usize), |(y, x)| {
        image.get_pixel(x as u32, y as u32)[0] as f32
    });
    let kernel_array: Array2<f32> = Array2::from_shape_fn((kh, kw), |(y, x)| kernel[y][x]);
    let output_array: Array2<f32> = Array2::from_shape_fn((output_height as usize, output_width as usize), |(y, x)| {
        input_array.slice(s![y..y + kh, x..x + kw]).iter().zip(kernel_array.iter()).map(|(a, b)| a * b).sum::<f32>().clamp(0.0, 255.0)
    });
    let mut output = ImageBuffer::new(output_width, output_height);
    Zip::indexed(&output_array).par_for_each(|(y, x), &val| {
        output.put_pixel(x as u32, y as u32, Luma([val as u8]));
    });
    output
}