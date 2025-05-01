use image::{GrayImage, ImageBuffer};
use cust::prelude::*;
use cust::blas::{CublasContext, Gemm, Operation};
use ndarray::{Array2, ArrayView2};

pub fn convolve(image: &GrayImage, kernel: &[&[f32]]) -> GrayImage {
    let (kh, kw) = (kernel.len(), kernel[0].len());
    let (width, height) = image.dimensions();
    let output_width = width - kw as u32 + 1;
    let output_height = height - kh as u32 + 1;
    let mut input_matrix = Array2::<f32>::zeros((output_height as usize * output_width as usize, kh * kw));
    for y in 0..output_height {
        for x in 0..output_width {
            let mut k = 0;
            while k + 3 < kh * kw {
                let ky0 = (k / kw) as u32;
                let kx0 = (k % kw) as u32;
                input_matrix[((y * output_width + x) as usize, k)] =
                    image.get_pixel(x + kx0, y + ky0)[0] as f32;
                let ky1 = ((k + 1) / kw) as u32;
                let kx1 = ((k + 1) % kw) as u32;
                input_matrix[((y * output_width + x) as usize, k + 1)] =
                    image.get_pixel(x + kx1, y + ky1)[0] as f32;
                let ky2 = ((k + 2) / kw) as u32;
                let kx2 = ((k + 2) % kw) as u32;
                input_matrix[((y * output_width + x) as usize, k + 2)] =
                    image.get_pixel(x + kx2, y + ky2)[0] as f32;
                let ky3 = ((k + 3) / kw) as u32;
                let kx3 = ((k + 3) % kw) as u32;
                input_matrix[((y * output_width + x) as usize, k + 3)] =
                    image.get_pixel(x + kx3, y + ky3)[0] as f32;
                k += 4;
            }
            while k < kh * kw {
                let ky = (k / kw) as u32;
                let kx = (k % kw) as u32;
                input_matrix[((y * output_width + x) as usize, k)] =
                    image.get_pixel(x + kx, y + ky)[0] as f32;
                k += 1;
            }
        }
    }
    let kernel_flat: Vec<f32> = kernel.iter().flatten().cloned().collect();
    let kernel_matrix = ArrayView2::from_shape((kh * kw, 1), &kernel_flat).unwrap();
    let output_size = (output_height * output_width) as usize;
    let mut output_data = vec![0.0f32; output_size];
    let _ctx = cust::quick_init().unwrap();
    let mut blas = CublasContext::new().unwrap();
    let d_a = DeviceBuffer::from_slice(input_matrix.as_slice().unwrap()).unwrap();
    let d_b = DeviceBuffer::from_slice(kernel_matrix.as_slice().unwrap()).unwrap();
    let mut d_c = DeviceBuffer::from_slice(&output_data).unwrap();
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