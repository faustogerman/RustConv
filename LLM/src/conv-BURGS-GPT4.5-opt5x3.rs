use image::{GrayImage, ImageBuffer, Luma};
use ndarray::{Array2, Ix2};
use cudarc::driver::{CudaDevice, LaunchConfig, DeviceRepr};
use cudarc::nvrtc::Ptx;
use std::error::Error;

const CUDA_KERNEL: &str = r#"
extern "C" __global__ void convolve_kernel(const float* image, const float* kernel, float* output, int img_width, int img_height, int kernel_width, int kernel_height, int out_width, int out_height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x < out_width && y < out_height) {
        float sum = 0.0;
        for(int ky = 0; ky < kernel_height; ky++) {
            for(int kx = 0; kx < kernel_width; kx++) {
                int ix = x + kx;
                int iy = y + ky;
                sum += image[iy * img_width + ix] * kernel[ky * kernel_width + kx];
            }
        }
        output[y * out_width + x] = sum;
    }
}
"#;

pub fn convolve(image: &GrayImage, kernel: &[&[f32]]) -> GrayImage {
    let dev = CudaDevice::new(0).unwrap();
    let ptx = Ptx::from_src(CUDA_KERNEL).unwrap();
    dev.load_ptx(ptx, "conv_module", &["convolve_kernel"]).unwrap();
    let kernel_fn = dev.get_kernel("conv_module", "convolve_kernel").unwrap();
    let (img_width, img_height) = image.dimensions();
    let (k_height, k_width) = (kernel.len(), kernel[0].len());
    let out_width = img_width - k_width as u32 + 1;
    let out_height = img_height - k_height as u32 + 1;
    let img_vec: Vec<f32> = image.pixels().map(|p| p[0] as f32).collect();
    let kernel_vec: Vec<f32> = kernel.iter().flat_map(|row| row.iter()).cloned().collect();
    let mut output_vec = vec![0f32; (out_width * out_height) as usize];
    let d_image = dev.alloc_and_copy(img_vec.as_slice()).unwrap();
    let d_kernel = dev.alloc_and_copy(kernel_vec.as_slice()).unwrap();
    let mut d_output = dev.alloc_zeros::<f32>(output_vec.len()).unwrap();
    let block_size = 16;
    let grid_x = (out_width as usize + block_size - 1) / block_size;
    let grid_y = (out_height as usize + block_size - 1) / block_size;
    let cfg = LaunchConfig {
        grid_dim: (grid_x as u32, grid_y as u32, 1),
        block_dim: (block_size as u32, block_size as u32, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        kernel_fn.launch(cfg, (&d_image, &d_kernel, &mut d_output, img_width as i32, img_height as i32, k_width as i32, k_height as i32, out_width as i32, out_height as i32)).unwrap();
    }
    dev.copy_to(&d_output, &mut output_vec).unwrap();
    let mut output_image = ImageBuffer::new(out_width, out_height);
    output_image.enumerate_pixels_mut().for_each(|(x, y, p)| {
        let val = output_vec[(y * out_width + x) as usize].clamp(0.0, 255.0) as u8;
        *p = Luma([val]);
    });
    output_image
}