use image::{GenericImage, GrayImage, ImageBuffer, Luma};
use rayon::prelude::*;

pub fn convolve(image: &GrayImage, kernel: &[&[f32]]) -> GrayImage {
    let (kh, kw) = (kernel.len(), kernel[0].len());
    let (width, height) = image.dimensions();
    let output_width = width - kw as u32 + 1;
    let output_height = height - kh as u32 + 1;
    let mut output = ImageBuffer::new(output_width, output_height);
    let block_size = 16u32;
    let max_y = output_height;
    let max_x = output_width;
    let image_data = image.as_raw();
    let kernel_flat: Vec<f32> = kernel.iter().flat_map(|row| row.iter()).copied().collect();
    let y_blocks: Vec<u32> = (0..max_y).step_by(block_size as usize).collect();
    y_blocks.into_par_iter().for_each(|yb| {
        let y_end = (yb + block_size).min(max_y);
        for y in yb..y_end {
            for xb in (0..max_x).step_by(block_size as usize) {
                let x_end = (xb + block_size).min(max_x);
                for x in xb..x_end {
                    let mut acc = 0.0;
                    let mut k = 0;
                    for ky in 0..kh {
                        let row_offset = (y + ky as u32) as usize * width as usize;
                        for kx in 0..kw {
                            acc += image_data[row_offset + (x + kx as u32) as usize] as f32 * kernel_flat[k];
                            k += 1;
                        }
                    }
                    unsafe {
                        output.unsafe_put_pixel(x, y, Luma([acc.clamp(0.0, 255.0) as u8]));
                    }
                }
            }
        }
    });
    output
}