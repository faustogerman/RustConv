use image::{GrayImage, ImageBuffer, Luma};
use cust::prelude::*;
use cust::blas::{CublasContext, Gemm, Operation};
use ndarray::{Array2, ArrayView2, ArrayViewMut2};

pub fn convolve(image: &GrayImage, kernel: &[&[f32]]) -> GrayImage {
    let (kh, kw) = (kernel.len(), kernel[0].len());
    let (width, height) = image.dimensions();
    let output_width = width - kw as u32 + 1;
    let output_height = height - kh as u32 + 1;
    let block_size = 32;
    let output_size = (output_width * output_height) as usize;
    let kernel_flat: Vec<f32> = kernel.iter().flatten().cloned().collect();
    let kernel_matrix = ArrayView2::from_shape((kh * kw, 1), &kernel_flat).unwrap();
    let mut output_data = vec![0.0f32; output_size];
    let _ctx = cust::quick_init().unwrap();
    let mut blas = CublasContext::new().unwrap();
    let d_kernel = DeviceBuffer::from_slice(kernel_matrix.as_slice().unwrap()).unwrap();
    for by in (0..output_height).step_by(block_size) {
        for bx in (0..output_width).step_by(block_size) {
            let block_h = (output_height - by).min(block_size);
            let block_w = (output_width - bx).min(block_size);
            let mut input_block = Array2::<f32>::zeros((block_h as usize * block_w as usize, kh * kw));
            for y in 0..block_h {
                for x in 0..block_w {
                    for ky in 0..kh {
                        for kx in 0..kw {
                            input_block[((y * block_w + x) as usize, ky * kw + kx)] =
                                image.get_pixel(bx + x + kx as u32, by + y + ky as u32)[0] as f32;
                        }
                    }
                }
            }
            let d_input = DeviceBuffer::from_slice(input_block.as_slice().unwrap()).unwrap();
            let mut block_output = vec![0.0f32; (block_h * block_w) as usize];
            let mut d_output = DeviceBuffer::from_slice(&block_output).unwrap();
            blas.gemm(
                Operation::N,
                Operation::N,
                (block_h * block_w) as i32,
                1,
                (kh * kw) as i32,
                &1.0f32,
                &d_input,
                (block_h * block_w) as i32,
                &d_kernel,
                (kh * kw) as i32,
                &0.0f32,
                &mut d_output,
                (block_h * block_w) as i32,
            ).unwrap();
            d_output.copy_to(&mut block_output).unwrap();
            for y in 0..block_h {
                for x in 0..block_w {
                    output_data[((by + y) * output_width + (bx + x)) as usize] =
                        block_output[(y * block_w + x) as usize];
                }
            }
        }
    }
    let output_image_data: Vec<u8> = output_data.iter().map(|&v| v.clamp(0.0, 255.0) as u8).collect();
    ImageBuffer::from_vec(output_width, output_height, output_image_data).unwrap()
}