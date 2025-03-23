use image::{GrayImage, ImageBuffer, Luma};
use cust::prelude::*;
use cust::blas::{CublasContext, Gemm, Operation};
use ndarray::{Array2, ArrayView2};
pub fn convolve(image: &GrayImage, kernel: &[&[f32]]) -> GrayImage {
    let (kh, kw) = (kernel.len(), kernel[0].len());
    let (width, height) = image.dimensions();
    let output_width = width - kw as u32 + 1;
    let output_height = height - kh as u32 + 1;
    let input_height = (output_height * output_width) as usize;
    let input_width = kh * kw;
    let mut input_matrix = Array2::<f32>::zeros((input_height, input_width));
    for by in (0..output_height).step_by(16) {
        for bx in (0..output_width).step_by(16) {
            let block_y_limit = (by + 16).min(output_height);
            let block_x_limit = (bx + 16).min(output_width);
            for y in by..block_y_limit {
                for x in bx..block_x_limit {
                    let row = (y * output_width + x) as usize;
                    for ky in 0..kh {
                        for kx in 0..kw {
                            input_matrix[(row, ky * kw + kx)] =
                                image.get_pixel(x + kx as u32, y + ky as u32)[0] as f32;
                        }
                    }
                }
            }
        }
    }
    let kernel_flat: Vec<f32> = kernel.iter().flatten().cloned().collect();
    let kernel_matrix = ArrayView2::from_shape((kh * kw, 1), &kernel_flat).unwrap();
    let mut output_data = vec![0.0f32; input_height];
    let _ctx = cust::quick_init().unwrap();
    let mut blas = CublasContext::new().unwrap();
    let d_a = DeviceBuffer::from_slice(input_matrix.as_slice().unwrap()).unwrap();
    let d_b = DeviceBuffer::from_slice(kernel_matrix.as_slice().unwrap()).unwrap();
    let mut d_c = DeviceBuffer::from_slice(&output_data).unwrap();
    blas.gemm(
        Operation::N,
        Operation::N,
        input_height as i32,
        1,
        input_width as i32,
        &1.0f32,
        &d_a,
        input_height as i32,
        &d_b,
        input_width as i32,
        &0.0f32,
        &mut d_c,
        input_height as i32,
    ).unwrap();
    d_c.copy_to(&mut output_data).unwrap();
    let output_image_data: Vec<u8> = output_data.iter().map(|&v| v.clamp(0.0, 255.0) as u8).collect();
    ImageBuffer::from_vec(output_width, output_height, output_image_data).unwrap()
}