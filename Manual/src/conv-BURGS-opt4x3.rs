extern crate image;
extern crate cust;
use image::{GrayImage, ImageBuffer};
use cust::prelude::*;
static CUDA_KERNEL_SRC: &str = r#"
extern "C" __global__ void convolve_gpu(const unsigned char *input, unsigned char *output, const float *kernel, int width, int height, int kw, int kh) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int output_width = width - kw + 1;
    int output_height = height - kh + 1;
    if (x < output_width && y < output_height) {
        float acc = 0.0f;
        int ky = 0;
        for (; ky <= kh - 4; ky += 4) {
            int kx = 0;
            for (; kx <= kw - 4; kx += 4) {
                int idx0 = (y + ky) * width + (x + kx);
                int idx1 = (y + ky) * width + (x + kx + 1);
                int idx2 = (y + ky) * width + (x + kx + 2);
                int idx3 = (y + ky) * width + (x + kx + 3);
                int idx4 = (y + ky + 1) * width + (x + kx);
                int idx5 = (y + ky + 1) * width + (x + kx + 1);
                int idx6 = (y + ky + 1) * width + (x + kx + 2);
                int idx7 = (y + ky + 1) * width + (x + kx + 3);
                int idx8 = (y + ky + 2) * width + (x + kx);
                int idx9 = (y + ky + 2) * width + (x + kx + 1);
                int idx10 = (y + ky + 2) * width + (x + kx + 2);
                int idx11 = (y + ky + 2) * width + (x + kx + 3);
                int idx12 = (y + ky + 3) * width + (x + kx);
                int idx13 = (y + ky + 3) * width + (x + kx + 1);
                int idx14 = (y + ky + 3) * width + (x + kx + 2);
                int idx15 = (y + ky + 3) * width + (x + kx + 3);
                int k_idx0 = ky * kw + kx;
                int k_idx1 = ky * kw + kx + 1;
                int k_idx2 = ky * kw + kx + 2;
                int k_idx3 = ky * kw + kx + 3;
                int k_idx4 = (ky + 1) * kw + kx;
                int k_idx5 = (ky + 1) * kw + kx + 1;
                int k_idx6 = (ky + 1) * kw + kx + 2;
                int k_idx7 = (ky + 1) * kw + kx + 3;
                int k_idx8 = (ky + 2) * kw + kx;
                int k_idx9 = (ky + 2) * kw + kx + 1;
                int k_idx10 = (ky + 2) * kw + kx + 2;
                int k_idx11 = (ky + 2) * kw + kx + 3;
                int k_idx12 = (ky + 3) * kw + kx;
                int k_idx13 = (ky + 3) * kw + kx + 1;
                int k_idx14 = (ky + 3) * kw + kx + 2;
                int k_idx15 = (ky + 3) * kw + kx + 3;
                acc += input[idx0] * kernel[k_idx0] + input[idx1] * kernel[k_idx1] + input[idx2] * kernel[k_idx2] + input[idx3] * kernel[k_idx3] +
                       input[idx4] * kernel[k_idx4] + input[idx5] * kernel[k_idx5] + input[idx6] * kernel[k_idx6] + input[idx7] * kernel[k_idx7] +
                       input[idx8] * kernel[k_idx8] + input[idx9] * kernel[k_idx9] + input[idx10] * kernel[k_idx10] + input[idx11] * kernel[k_idx11] +
                       input[idx12] * kernel[k_idx12] + input[idx13] * kernel[k_idx13] + input[idx14] * kernel[k_idx14] + input[idx15] * kernel[k_idx15];
            }
            for (; kx < kw; kx++) {
                acc += input[(y + ky) * width + (x + kx)] * kernel[ky * kw + kx];
                acc += input[(y + ky + 1) * width + (x + kx)] * kernel[(ky + 1) * kw + kx];
                acc += input[(y + ky + 2) * width + (x + kx)] * kernel[(ky + 2) * kw + kx];
                acc += input[(y + ky + 3) * width + (x + kx)] * kernel[(ky + 3) * kw + kx];
            }
        }
        for (; ky < kh; ky++) {
            for (int kx = 0; kx < kw; kx++) {
                acc += input[(y + ky) * width + (x + kx)] * kernel[ky * kw + kx];
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
    let block_size = (16, 16, 1);
    let grid_size = ((output_width + block_size.0 - 1) / block_size.0, (output_height + block_size.1 - 1) / block_size.1, 1);
    let module = Module::from_ptx(CUDA_KERNEL_SRC, &[]).unwrap();
    let stream = Stream::new(StreamFlags::DEFAULT, None).unwrap();
    let input_flat: Vec<u8> = image.to_vec();
    let kernel_flat: Vec<f32> = kernel.iter().flat_map(|row| row.iter()).cloned().collect();
    let mut output_flat = vec![0u8; (output_width * output_height) as usize];
    let input_gpu = DeviceBuffer::from_slice(&input_flat).unwrap();
    let kernel_gpu = DeviceBuffer::from_slice(&kernel_flat).unwrap();
    let mut output_gpu = DeviceBuffer::from_slice(&output_flat).unwrap();
    let func = module.get_function("convolve_gpu").unwrap();
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