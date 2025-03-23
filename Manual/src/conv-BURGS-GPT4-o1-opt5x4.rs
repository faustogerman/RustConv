use image::{GrayImage, ImageBuffer, Luma};
use rayon::prelude::*;

pub fn convolve(image: &GrayImage, kernel: &[&[f32]]) -> GrayImage {
    let (kh, kw) = (kernel.len(), kernel[0].len());
    let (width, height) = image.dimensions();
    let output_width = width - kw as u32 + 1;
    let output_height = height - kh as u32 + 1;
    let flatten_kernel: Vec<f32> = kernel.iter().flatten().copied().collect();
    let data = image.as_raw();
    let mut buffer = vec![0u8; (output_width * output_height) as usize];
    buffer
        .par_chunks_mut(output_width as usize)
        .enumerate()
        .for_each(|(y, row)| {
            let y_u32 = y as u32;
            for x in 0..output_width {
                let mut acc = 0.0;
                let mut i = 0;
                for ky in 0..kh {
                    let row_offset = (y_u32 + ky as u32) * width;
                    for kx in 0..kw {
                        let pixel_value = data[(row_offset + x + kx as u32) as usize] as f32;
                        acc += pixel_value * flatten_kernel[i];
                        i += 1;
                    }
                }
                row[x as usize] = acc.clamp(0.0, 255.0) as u8;
            }
        });
    let mut output = ImageBuffer::new(output_width, output_height);
    for (i, &pixel) in buffer.iter().enumerate() {
        let x = (i % output_width as usize) as u32;
        let y = (i / output_width as usize) as u32;
        output.put_pixel(x, y, Luma([pixel]));
    }
    output
}