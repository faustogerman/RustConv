use image::{GrayImage, ImageBuffer, Luma};
use ndarray::{Array2, s};
use ndarray::parallel::prelude::*;
use ndarray_conv::Conv2D;
use cust::prelude::*;
use cust::memory::*;
use std::error::Error;

pub fn convolve(image: &GrayImage, kernel: &[&[f32]]) -> GrayImage {
    let ctx = cust::quick_init().unwrap();
    let stream = Stream::new(StreamFlags::DEFAULT, None).unwrap();
    let ptx = "
    extern \"C\" __global__ void conv2d_gpu(const float* img, const float* kernel, float* output, int img_w, int img_h, int kernel_w, int kernel_h, int out_w, int out_h) {
        int x = blockDim.x * blockIdx.x + threadIdx.x;
        int y = blockDim.y * blockIdx.y + threadIdx.y;
        if (x < out_w && y < out_h) {
            float sum = 0.0;
            for (int ky = 0; ky < kernel_h; ky++) {
                for (int kx = 0; kx < kernel_w; kx++) {
                    int ix = x + kx;
                    int iy = y + ky;
                    sum += img[iy * img_w + ix] * kernel[ky * kernel_w + kx];
                }
            }
            output[y * out_w + x] = sum;
        }
    }";
    let module = Module::from_ptx(ptx, &[]).unwrap();
    let func = module.get_function("conv2d_gpu").unwrap();
    let (width, height) = image.dimensions();
    let img_w = width as usize;
    let img_h = height as usize;
    let kernel_w = kernel[0].len();
    let kernel_h = kernel.len();
    let out_w = img_w - kernel_w + 1;
    let out_h = img_h - kernel_h + 1;
    let img_array: Vec<f32> = image.pixels().map(|p| p[0] as f32).collect();
    let kernel_array: Vec<f32> = kernel.iter().flat_map(|&row| row.iter()).copied().collect();
    let mut output_array = vec![0f32; out_w * out_h];
    let d_img = DeviceBuffer::from_slice(&img_array).unwrap();
    let d_kernel = DeviceBuffer::from_slice(&kernel_array).unwrap();
    let mut d_output = DeviceBuffer::from_slice(&output_array).unwrap();
    let device = Device::get_device(0).unwrap();
    let max_threads = device.get_attribute(DeviceAttribute::MaxThreadsPerBlock).unwrap() as u32;
    let block_size_x = (max_threads as f32).sqrt() as u32;
    let block_size_y = block_size_x;
    let grid_size_x = ((out_w as u32 + block_size_x - 1) / block_size_x) as u32;
    let grid_size_y = ((out_h as u32 + block_size_y - 1) / block_size_y) as u32;
    unsafe {
        launch!(func<<<(grid_size_x, grid_size_y, 1), (block_size_x, block_size_y, 1), 0, stream>>>(
            d_img.as_device_ptr(),
            d_kernel.as_device_ptr(),
            d_output.as_device_ptr(),
            img_w as i32,
            img_h as i32,
            kernel_w as i32,
            kernel_h as i32,
            out_w as i32,
            out_h as i32
        )).unwrap();
    }
    stream.synchronize().unwrap();
    d_output.copy_to(&mut output_array).unwrap();
    let mut output = ImageBuffer::new(out_w as u32, out_h as u32);
    output.enumerate_pixels_mut().for_each(|(x, y, pixel)| {
        let val = output_array[y as usize * out_w + x as usize].clamp(0.0, 255.0) as u8;
        *pixel = Luma([val]);
    });
    output
}