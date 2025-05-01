use image::{GrayImage, ImageBuffer, Luma};
use cust::prelude::*;
use cust::blas::{CublasContext, Gemm, Operation};
use ndarray::{Array2, ArrayView2};
use std::error::Error;

pub fn convolve(image: &GrayImage, kernel: &[&[f32]]) -> GrayImage {
    let (kh, kw) = (kernel.len(), kernel[0].len());
    let (width, height) = image.dimensions();
    let output_width = width - kw as u32 + 1;
    let output_height = height - kh as u32 + 1;
    let input_size = (output_height * output_width) as usize;
    let kernel_size = kh * kw;
    let mut input_matrix = Array2::<f32>::zeros((kernel_size, input_size));
    for y in 0..output_height {
        for x in 0..output_width {
            let col = (y * output_width + x) as usize;
            for ky in 0..kh {
                for kx in 0..kw {
                    input_matrix[(ky * kw + kx, col)] =
                        image.get_pixel(x + kx as u32, y + ky as u32)[0] as f32;
                }
            }
        }
    }
    let kernel_flat: Vec<f32> = kernel.iter().flatten().cloned().collect();
    let kernel_matrix = ArrayView2::from_shape((1, kernel_size), &kernel_flat).unwrap();
    let ctx = cust::quick_init().unwrap();
    let mut blas = CublasContext::new().unwrap();
    let d_input = DeviceBuffer::from_slice(input_matrix.as_slice().unwrap()).unwrap();
    let d_kernel = DeviceBuffer::from_slice(kernel_matrix.as_slice().unwrap()).unwrap();
    let mut d_output = DeviceBuffer::<f32>::zeroed(input_size).unwrap();
    let algo = cust::blas::GemmAlgo::default();
    blas.gemm_ex(
        Operation::N,
        Operation::N,
        1,
        input_size as i32,
        kernel_size as i32,
        &1.0f32,
        &d_kernel,
        cudaDataType::CUDA_R_32F,
        1,
        &d_input,
        cudaDataType::CUDA_R_32F,
        kernel_size as i32,
        &0.0f32,
        &mut d_output,
        cudaDataType::CUDA_R_32F,
        1,
        cudaDataType::CUDA_R_32F,
        algo,
    ).unwrap();
    let mut output_data = vec![0.0f32; input_size];
    d_output.copy_to(&mut output_data).unwrap();
    let output_image_data: Vec<u8> = output_data.iter().map(|&v| v.clamp(0.0, 255.0) as u8).collect();
    ImageBuffer::from_vec(output_width, output_height, output_image_data).unwrap()
}