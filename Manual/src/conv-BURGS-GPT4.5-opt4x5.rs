extern crate image;
extern crate cust;
use image::{GrayImage, ImageBuffer};
use cust::prelude::*;
use cust::memory::*;
static CUDA_KERNEL_SRC: &str = r#"
extern "C"
__global__ void convolve_gpu(const unsigned char* input, unsigned char* output, const float* kernel, int width, int height, int kw, int kh) {
    extern __shared__ unsigned char shared_mem[];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;
    int output_width = width - kw + 1;
    int output_height = height - kh + 1;
    int shared_width = blockDim.x + kw - 1;
    int shared_height = blockDim.y + kh - 1;
    int shared_x = tx;
    int shared_y = ty;
    for (int dy = shared_y; dy < shared_height; dy += blockDim.y) {
        for (int dx = shared_x; dx < shared_width; dx += blockDim.x) {
            int global_x = blockIdx.x * blockDim.x + dx;
            int global_y = blockIdx.y * blockDim.y + dy;
            if (global_x < width && global_y < height) {
                shared_mem[dy * shared_width + dx] = input[global_y * width + global_x];
            } else {
                shared_mem[dy * shared_width + dx] = 0;
            }
        }
    }
    __syncthreads();
    if (x < output_width && y < output_height) {
        float acc = 0.0f;
        for (int ky = 0; ky < kh; ky++) {
            for (int kx = 0; kx < kw; kx++) {
                acc += shared_mem[(ty + ky) * shared_width + (tx + kx)] * kernel[ky * kw + kx];
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
    let ctx = cust::quick_init().unwrap();
    let input_flat = image.as_raw();
    let kernel_flat: Vec<f32> = kernel.iter().flat_map(|row| row.iter()).cloned().collect();
    let mut output_flat = vec![0u8; (output_width * output_height) as usize];
    let module = Module::from_ptx(CUDA_KERNEL_SRC, &[]).unwrap();
    let stream = Stream::new(StreamFlags::DEFAULT, None).unwrap();
    let input_gpu = DeviceBuffer::from_slice(input_flat).unwrap();
    let kernel_gpu = DeviceBuffer::from_slice(&kernel_flat).unwrap();
    let mut output_gpu = DeviceBuffer::from_slice(&output_flat).unwrap();
    let func = module.get_function("convolve_gpu").unwrap();
    let block_dim = (16u32, 16u32, 1u32);
    let grid_dim = (
        (output_width + block_dim.0 - 1) / block_dim.0,
        (output_height + block_dim.1 - 1) / block_dim.1,
        1u32,
    );
    let shared_mem_size = ((block_dim.0 + kw as u32 - 1) * (block_dim.1 + kh as u32 - 1)) as usize;
    unsafe {
        launch!(
            func<<<grid_dim, block_dim, shared_mem_size, stream>>>(
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