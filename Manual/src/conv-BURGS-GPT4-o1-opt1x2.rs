use image::{GrayImage, ImageBuffer, Luma};
use std::time::Instant;

fn run_convolution(image: &GrayImage, kernel: &[&[f32]], block_size: u32) -> ImageBuffer<Luma<u8>, Vec<u8>> {
    let (kh, kw) = (kernel.len(), kernel[0].len());
    let (width, height) = image.dimensions();
    let output_width = width - kw as u32 + 1;
    let output_height = height - kh as u32 + 1;
    let mut output = ImageBuffer::new(output_width, output_height);
    let max_y = output_height;
    let max_x = output_width;
    for yb in (0..max_y).step_by(block_size as usize) {
        for xb in (0..max_x).step_by(block_size as usize) {
            let y_end = (yb + block_size).min(max_y);
            let x_end = (xb + block_size).min(max_x);
            for y in yb..y_end {
                for x in xb..x_end {
                    let mut acc = 0.0;
                    for ky in 0..kh {
                        for kx in 0..kw {
                            let pv = image.get_pixel(x + kx as u32, y + ky as u32)[0] as f32;
                            acc += pv * kernel[ky][kx];
                        }
                    }
                    output.put_pixel(x, y, Luma([(acc.clamp(0.0, 255.0)) as u8]));
                }
            }
        }
    }
    output
}

pub fn convolve(image: &GrayImage, kernel: &[&[f32]]) -> GrayImage {
    let block_sizes = [4, 8, 16, 32];
    let mut best_time = std::time::Duration::MAX;
    let mut best_block_size = block_sizes[0];
    for &bsize in &block_sizes {
        let start = Instant::now();
        let _ = run_convolution(image, kernel, bsize);
        let elapsed = start.elapsed();
        if elapsed < best_time {
            best_time = elapsed;
            best_block_size = bsize;
        }
    }
    run_convolution(image, kernel, best_block_size)
}