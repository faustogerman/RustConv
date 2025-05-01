use image::{GrayImage, ImageBuffer};
use cust::prelude::*;
use std::error::Error;
use cust::blas::{CublasContext, Gemm, Operation};
use ndarray::{Array2, ArrayView2};
use cust::device::Device;
use cust::context::ContextFlags;

fn get_device_properties() -> Result<(i32, i32), Box<dyn Error>> {
    let device = Device::get_device(0)?;
    let props = device.get_attributes()?;
    let max_threads_per_block = props.max_threads_per_block;
    let multiprocessor_count = props.multiprocessor_count;
    Ok((max_threads_per_block, multiprocessor_count))
}

pub fn convolve(image: &GrayImage, kernel: &[&[f32]]) -> GrayImage {
    let (kh, kw) = (kernel.len(), kernel[0].len());
    let (width, height) = image.dimensions();
    let output_width = width - kw as u32 + 1;
    let output_height = height - kh as u32 + 1;
    let mut input_matrix = Array2::<f32>::zeros((output_height as usize * output_width as usize, kh * kw));
    for y in 0..output_height {
        for x in 0..output_width {
            for ky in 0..kh {
                for kx in 0..kw {
                    input_matrix[((y * output_width + x) as usize, ky * kw + kx)] =
                        image.get_pixel(x + kx as u32, y + ky as u32)[0] as f32;
                }
            }
        }
    }
    let kernel_flat: Vec<f32> = kernel.iter().flatten().cloned().collect();
    let kernel_matrix = ArrayView2::from_shape((kh * kw, 1), &kernel_flat).unwrap();
    let output_size = (output_height * output_width) as usize;
    let mut output_data = vec![0.0f32; output_size];
    let _ctx = cust::quick_init().unwrap();
    let (max_threads_per_block, multiprocessor_count) = get_device_properties().unwrap();
    let optimal_block_size = (max_threads_per_block as usize).min(1024);
    let grid_size = ((output_size + optimal_block_size - 1) / optimal_block_size).min(multiprocessor_count as usize * 32);
    let mut blas = CublasContext::new().unwrap();
    let d_a = DeviceBuffer::from_slice(input_matrix.as_slice().unwrap()).unwrap();
    let d_b = DeviceBuffer::from_slice(kernel_matrix.as_slice().unwrap()).unwrap();
    let mut d_c = DeviceBuffer::from_slice(&output_data).unwrap();
    blas.set_stream(Stream::new(StreamFlags::NON_BLOCKING, None).unwrap()).unwrap();
    blas.gemm(
        Operation::N,
        Operation::N,
        output_size as i32,
        1,
        (kh * kw) as i32,
        &1.0f32,
        &d_a,
        output_size as i32,
        &d_b,
        (kh * kw) as i32,
        &0.0f32,
        &mut d_c,
        output_size as i32,
    ).unwrap();
    d_c.copy_to(&mut output_data).unwrap();
    let output_image_data: Vec<u8> = output_data.iter().map(|&v| v.clamp(0.0, 255.0) as u8).collect();
    ImageBuffer::from_vec(output_width, output_height, output_image_data).unwrap()
}