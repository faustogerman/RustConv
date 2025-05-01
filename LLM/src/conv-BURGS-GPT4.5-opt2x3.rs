use image::{GrayImage, ImageBuffer, Luma};
use ndarray::{Array2, s, Zip};
use cudarc::driver::{CudaDevice, LaunchConfig};
use cudarc::nvrtc::Ptx;
use cudarc::driver::DeviceRepr;

const KERNEL_SRC: &str = "
extern \"C\" __global__ void convolve_kernel(const float* __restrict__ img, const float* __restrict__ kernel, float* __restrict__ output, int img_w, int img_h, int kernel_w, int kernel_h, int output_w, int output_h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= output_w || y >= output_h) return;
    float sum = 0.0f;
    for (int ky = 0; ky < kernel_h; ++ky) {
        for (int kx = 0; kx < kernel_w; ++kx) {
            int ix = x + kx;
            int iy = y + ky;
            sum += img[iy * img_w + ix] * kernel[ky * kernel_w + kx];
        }
    }
    output[y * output_w + x] = fminf(fmaxf(sum, 0.0f), 255.0f);
}
";

pub fn convolve(image: &GrayImage, kernel: &[&[f32]]) -> GrayImage {
    let device = CudaDevice::new(0).unwrap();
    let ptx = Ptx::from_src(KERNEL_SRC).unwrap();
    device.load_ptx(ptx, "convolve_kernel", &["convolve_kernel"]).unwrap();
    let (kh, kw) = (kernel.len(), kernel[0].len());
    let kernel_flat: Vec<f32> = kernel.iter().flat_map(|r| r.iter()).cloned().collect();
    let (width, height) = image.dimensions();
    let output_width = width - kw as u32 + 1;
    let output_height = height - kh as u32 + 1;
    let img_arr_host: Vec<f32> = image.pixels().map(|p| p[0] as f32).collect();
    let mut output_host = vec![0.0f32; (output_width * output_height) as usize];
    let img_dev = device.htod_copy(&img_arr_host).unwrap();
    let kernel_dev = device.htod_copy(&kernel_flat).unwrap();
    let mut output_dev = device.alloc_zeros::<f32>(output_host.len()).unwrap();
    let threads_x = 16;
    let threads_y = 16;
    let blocks_x = ((output_width + threads_x - 1) / threads_x) as u32;
    let blocks_y = ((output_height + threads_y - 1) / threads_y) as u32;
    let cfg = LaunchConfig {
        grid_dim: (blocks_x, blocks_y, 1),
        block_dim: (threads_x, threads_y, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        device.launch(
            "convolve_kernel",
            cfg,
            (
                img_dev.as_device_ptr(),
                kernel_dev.as_device_ptr(),
                output_dev.as_device_ptr(),
                width as i32,
                height as i32,
                kw as i32,
                kh as i32,
                output_width as i32,
                output_height as i32,
            ),
        ).unwrap();
    }
    device.dtoh_copy_into(&output_dev, &mut output_host).unwrap();
    let mut output = ImageBuffer::new(output_width, output_height);
    for y in 0..output_height as usize {
        for x in 0..output_width as usize {
            let val = output_host[y * output_width as usize + x];
            output.put_pixel(x as u32, y as u32, Luma([val as u8]));
        }
    }
    output
}