use image::{GrayImage, ImageBuffer, Luma};
use ndarray::{Array2, s};
use ndarray::Zip;
use blas::{sgemm, Transpose};

extern crate blas;

pub fn convolve(image: &GrayImage, kernel: &[&[f32]]) -> GrayImage {
    let (kh, kw) = (kernel.len(), kernel[0].len());
    let (width, height) = image.dimensions();
    let output_width = width - kw as u32 + 1;
    let output_height = height - kh as u32 + 1;
    let input_array: Array2<f32> = Array2::from_shape_fn((height as usize, width as usize), |(y, x)| {
        image.get_pixel(x as u32, y as u32)[0] as f32
    });
    let kernel_array: Array2<f32> = Array2::from_shape_fn((kh, kw), |(y, x)| kernel[kh - 1 - y][kw - 1 - x]);
    let mut col_matrix = Array2::<f32>::zeros((kw * kh, (output_width * output_height) as usize));
    for y in 0..output_height as usize {
        for x in 0..output_width as usize {
            let window = input_array.slice(s![y..y + kh, x..x + kw]);
            let col = window.iter().cloned().collect::<Vec<f32>>();
            let col_index = y * output_width as usize + x;
            col_matrix.column_mut(col_index).assign(&Array2::from_shape_vec((kw * kh, 1), col).unwrap().column(0));
        }
    }
    let kernel_flat = kernel_array.iter().cloned().collect::<Vec<f32>>();
    let mut result = vec![0f32; (output_width * output_height) as usize];
    unsafe {
        sgemm(
            Transpose::None,
            Transpose::None,
            1,
            (output_width * output_height) as i32,
            (kw * kh) as i32,
            1.0,
            &kernel_flat,
            1,
            &col_matrix.as_slice().unwrap(),
            (kw * kh) as i32,
            0.0,
            &mut result,
            1,
        );
    }
    let mut output = ImageBuffer::new(output_width, output_height);
    Zip::indexed(&Array2::from_shape_vec((output_height as usize, output_width as usize), result).unwrap()).for_each(|(y, x), &val| {
        output.put_pixel(x as u32, y as u32, Luma([val.clamp(0.0, 255.0) as u8]));
    });
    output
}