use image::{GrayImage, ImageBuffer, Luma};
use ndarray::{Array2, s, Zip};
use cust::prelude::*;
use std::error::Error;

static PTX: &str = r#"
extern "C" __global__ void convolve_kernel(
    const float* input, const float* kernel, float* output, 
    int width, int height, int kw, int kh, int output_width, int output_height
) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    if (tx < output_width && ty < output_height) {
        float sum = 0.0;
        for (int ky = 0; ky < kh; ky++) {
            for (int kx = 0; kx < kw; kx++) {
                sum += input[(ty + ky) * width + (tx + kx)] * kernel[ky * kw + kx];
            }
        }
        sum = fminf(fmaxf(sum, 0.0), 255.0);
        output[ty * output_width + tx] = sum;
    }
}
"#;

pub fn convolve(image: &GrayImage, kernel: &[&[f32]]) -> GrayImage {
    let (kh, kw) = (kernel.len(), kernel[0].len());
    let (width, height) = image.dimensions();
    let output_width = width - kw as u32 + 1;
    let output_height = height - kh as u32 + 1;
    let input_array: Vec<f32> = image.pixels().map(|p| p[0] as f32).collect();
    let kernel_array: Vec<f32> = kernel.iter().flat_map(|row| row.iter()).cloned().collect();
    let output_len = (output_width * output_height) as usize;
    let mut output_array = vec![0f32; output_len];
    let _ = cust::quick_init();
    let ctx = cust::context::CurrentContext::get_current().unwrap();
    let module = Module::from_ptx(PTX, &[]).unwrap();
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();
    let input_gpu = DeviceBuffer::from_slice(&input_array).unwrap();
    let kernel_gpu = DeviceBuffer::from_slice(&kernel_array).unwrap();
    let mut output_gpu = DeviceBuffer::from_slice(&output_array).unwrap();
    let block_size = (16, 16, 1);
    let grid_size = (
        (output_width as usize + block_size.0 - 1) / block_size.0,
        (output_height as usize + block_size.1 - 1) / block_size.1,
        1,
    );
    let func = module.get_function("convolve_kernel").unwrap();
    unsafe {
        launch!(func<<<grid_size, block_size, 0, stream>>>(
            input_gpu.as_device_ptr(),
            kernel_gpu.as_device_ptr(),
            output_gpu.as_device_ptr(),
            width as i32,
            height as i32,
            kw as i32,
            kh as i32,
            output_width as i32,
            output_height as i32
        )).unwrap();
    }
    stream.synchronize().unwrap();
    output_gpu.copy_to(&mut output_array).unwrap();
    let mut output = ImageBuffer::new(output_width, output_height);
    Zip::indexed(&Array2::from_shape_vec((output_height as usize, output_width as usize), output_array).unwrap())
        .for_each(|(y, x), &val| {
            output.put_pixel(x as u32, y as u32, Luma([val as u8]));
        });
    output
}