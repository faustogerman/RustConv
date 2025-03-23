use image::{GrayImage, ImageBuffer, Luma};
use rayon::prelude::*;

pub fn convolve(image: &GrayImage, kernel: &[&[f32]]) -> GrayImage {
    let block_size: usize = 16;
    let (kh, kw) = (kernel.len(), kernel[0].len());
    let (width, height) = image.dimensions();
    let output_width = width - kw as u32 + 1;
    let output_height = height - kh as u32 + 1;
    let mut buffer = vec![0u8; (output_width * output_height) as usize];
    buffer
        .par_chunks_mut(output_width as usize)
        .enumerate()
        .for_each(|(y, row)| {
            let y_u32 = y as u32;
            for x_block in (0..output_width as usize).step_by(block_size) {
                let end_x = (x_block + block_size).min(output_width as usize);
                for x in x_block..end_x {
                    let x_u32 = x as u32;
                    let mut acc = 0.0;
                    for ky in 0..kh {
                        for kx in 0..kw {
                            let pixel_value = image.get_pixel(x_u32 + kx as u32, y_u32 + ky as u32)[0] as f32;
                            acc += pixel_value * kernel[ky][kx];
                        }
                    }
                    row[x] = acc.clamp(0.0, 255.0) as u8;
                }
            }
        });
    let mut output = ImageBuffer::new(output_width, output_height);
    for (i, pixel) in buffer.iter().enumerate() {
        let x = (i % output_width as usize) as u32;
        let y = (i / output_width as usize) as u32;
        output.put_pixel(x, y, Luma([*pixel]));
    }
    output
}