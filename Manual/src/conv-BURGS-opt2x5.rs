use image::{GrayImage, ImageBuffer, Luma};
use ndarray::{Array2, s};
use ndarray::parallel::prelude::*;
use rayon::prelude::*;

pub fn convolve(image: &GrayImage, kernel: &[&[f32]]) -> GrayImage {
    let (kh, kw) = (kernel.len(), kernel[0].len());
    let kernel: Array2<f32> = Array2::from_shape_vec((kh, kw), kernel.iter().flat_map(|r| r.iter()).cloned().collect()).unwrap();
    let (width, height) = image.dimensions();
    let output_width = width - kw as u32 + 1;
    let output_height = height - kh as u32 + 1;
    let img_arr: Array2<f32> = Array2::from_shape_fn((height as usize, width as usize), |(y, x)| image.get_pixel(x as u32, y as u32)[0] as f32);
    let mut output_arr: Array2<f32> = Array2::zeros((output_height as usize, output_width as usize));
    output_arr.axis_chunks_iter_mut(ndarray::Axis(0), 32).into_par_iter().enumerate().for_each(|(by, mut chunk)| {
        let y_start = by * 32;
        let y_end = y_start + chunk.len_of(ndarray::Axis(0));
        for (y, mut row) in (y_start..y_end).zip(chunk.axis_iter_mut(ndarray::Axis(0))) {
            for x in 0..output_width as usize {
                let window = img_arr.slice(s![y..y + kh, x..x + kw]);
                let acc = (&window * &kernel).sum();
                row[x] = acc.clamp(0.0, 255.0);
            }
        }
    });
    let output_vec: Vec<u8> = output_arr.iter().map(|&val| val as u8).collect();
    ImageBuffer::from_raw(output_width, output_height, output_vec).unwrap()
}