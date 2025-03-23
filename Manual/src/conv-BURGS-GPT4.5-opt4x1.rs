extern crate image;
extern crate cust;
use image::{GrayImage, ImageBuffer};
use cust::prelude::*;
static CUDA_KERNEL_SRC: &str = r#"
#define BLOCK_SIZE 16
extern "C" __global__ void convolve_gpu(const unsigned char *input, unsigned char *output, const float *kernel, int width, int height, int kw, int kh) {
    __shared__ unsigned char shared_input[BLOCK_SIZE + 32][BLOCK_SIZE + 32];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * BLOCK_SIZE + tx;
    int y = blockIdx.y * BLOCK_SIZE + ty;
    int output_width = width - kw + 1;
    int output_height = height - kh + 1;
    int shared_size_x = BLOCK_SIZE + kw - 1;
    int shared_size_y = BLOCK_SIZE + kh - 1;
    for (int j = ty; j < shared_size_y; j += BLOCK_SIZE) {
        for (int i = tx; i < shared_size_x; i += BLOCK_SIZE) {
            int global_x = blockIdx.x * BLOCK_SIZE + i;
            int global_y = blockIdx.y * BLOCK_SIZE + j;
            if (global_x < width && global_y < height) {
                shared_input[j][i] = input[global_y * width + global_x];
            } else {
                shared_input[j][i] = 0;
            }
        }
    }
    __syncthreads();
    if (x < output_width && y < output_height) {
        float acc = 0.0f;
        for (int ky = 0; ky < kh; ky++) {
            for (int kx = 0; kx < kw; kx++) {
                acc += shared_input[ty + ky][tx + kx] * kernel[ky * kw + kx];
            }
        }
        acc = fminf(fmaxf(acc, 0.0f), 255.0f);
        output[y * output_width + x] = (unsigned char)(acc);
    }
}
"#;
pub fn convolve(image: &GrayImage, kernel: &[&[f32]]) -> GrayImage {
    let (kw, kh) = (kernel[0].len(), kernel.len());
    let (width, height) = image.dimensions();
    let output_width = width - kw as u32 + 1;
    let output_height = height - kh as u32 + 1;
    let _ctx = cust::quick_init().unwrap();
    let module = Module::from_ptx(CUDA_KERNEL_SRC, &[]).unwrap();
    let stream = Stream::new(StreamFlags::DEFAULT, None).unwrap();
    let input_flat: Vec<u8> = image.to_vec();
    let kernel_flat: Vec<f32> = kernel.iter().flat_map(|row| row.iter()).cloned().collect();
    let mut output_flat = vec![0u8; (output_width * output_height) as usize];
    let input_gpu = DeviceBuffer::from_slice(&input_flat).unwrap();
    let kernel_gpu = DeviceBuffer::from_slice(&kernel_flat).unwrap();
    let mut output_gpu = DeviceBuffer::from_slice(&output_flat).unwrap();
    let func = module.get_function("convolve_gpu").unwrap();
    let block_size = (16, 16, 1);
    let grid_size = (
        (output_width + block_size.0 - 1) / block_size.0,
        (output_height + block_size.1 - 1) / block_size.1,
        1,
    );
    unsafe {
        launch!(
            func<<<grid_size, block_size, 0, stream>>>(
                input_gpu.as_device_ptr(),
                output_gpu.as_device_ptr(),
                kernel_gpu.as_device_ptr(),
                width as i32,
                height as i32,
                kw as i32,
                kh as i32
            )
        )
        .unwrap();
    }
    stream.synchronize().unwrap();
    output_gpu.copy_to(&mut output_flat).unwrap();
    ImageBuffer::from_vec(output_width, output_height, output_flat).unwrap()
}