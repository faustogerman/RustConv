use image::{GrayImage, ImageBuffer, Luma};
use ndarray::{Array2, s};
use ndarray::Zip;
use ndarray::linalg::general_mat_mul;
use ndarray::ShapeBuilder;

pub fn convolve(image: &GrayImage, kernel: &[&[f32]]) -> GrayImage {
    let (kh, kw) = (kernel.len(), kernel[0].len());
    let (width, height) = image.dimensions();
    let output_width = width - kw as u32 + 1;
    let output_height = height - kh as u32 + 1;
    let input_array: Array2<f32> = Array2::from_shape_fn((height as usize, width as usize), |(y, x)| {
        image.get_pixel(x as u32, y as u32)[0] as f32
    });
    let kernel_array: Array2<f32> = Array2::from_shape_fn((kh, kw), |(y, x)| kernel[kh - 1 - y][kw - 1 - x]);
    let mut im2col = Array2::<f32>::zeros((kh * kw, (output_height * output_width) as usize));
    for y in 0..output_height as usize {
        for x in 0..output_width as usize {
            let window = input_array.slice(s![y..y + kh, x..x + kw]).iter().cloned();
            let idx = y * output_width as usize + x;
            im2col.column_mut(idx).assign(&Array2::from_shape_vec((kh * kw, 1), window.collect()).unwrap().column(0));
        }
    }
    let kernel_flat = kernel_array.into_shape((1, kh * kw)).unwrap();
    let mut conv_result = Array2::<f32>::zeros((1, (output_height * output_width) as usize));
    general_mat_mul(1.0, &kernel_flat, &im2col, 0.0, &mut conv_result);
    let mut output = ImageBuffer::new(output_width, output_height);
    Zip::indexed(conv_result.into_shape((output_height as usize, output_width as usize)).unwrap()).for_each(|(y, x), &val| {
        output.put_pixel(x as u32, y as u32, Luma([val.clamp(0.0, 255.0) as u8]));
    });
    output
}