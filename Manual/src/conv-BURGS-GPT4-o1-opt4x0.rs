use image::{GrayImage, ImageBuffer, Luma};
use rayon::prelude::*;

pub fn convolve(image: &GrayImage, kernel: &[&[f32]]) -> GrayImage {
    let (kh, kw) = (kernel.len(), kernel[0].len());
    let (width, height) = image.dimensions();
    let output_width = width - kw as u32 + 1;
    let output_height = height - kh as u32 + 1;
    let mut data = vec![0u8; (output_width * output_height) as usize];

    (0..output_height).into_par_iter().for_each(|y| {
        for x in 0..output_width {
            let mut acc = 0.0;
            for ky in 0..kh {
                for kx in 0..kw {
                    let pixel_value = image.get_pixel(x + kx as u32, y + ky as u32)[0] as f32;
                    let weight = kernel[ky][kx];
                    acc += pixel_value * weight;
                }
            }
            data[(y * output_width + x) as usize] = acc.clamp(0.0, 255.0) as u8;
        }
    });

    let mut output = ImageBuffer::new(output_width, output_height);
    for (i, &val) in data.iter().enumerate() {
        let x = i as u32 % output_width;
        let y = i as u32 / output_width;
        output.put_pixel(x, y, Luma([val]));
    }
    output
}