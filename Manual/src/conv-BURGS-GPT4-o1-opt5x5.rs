use image::{GrayImage, ImageBuffer, Luma};
use rayon::prelude::*;
use matrixmultiply::sgemm;

pub fn convolve(image: &GrayImage, kernel: &[&[f32]]) -> GrayImage {
    let (kh, kw) = (kernel.len(), kernel[0].len());
    let (width, height) = image.dimensions();
    let output_width = width - kw as u32 + 1;
    let output_height = height - kh as u32 + 1;
    let m = (output_width * output_height) as usize;
    let k = (kh * kw) as usize;
    let n = 1;
    let mut im2col = vec![0.0f32; m * k];
    im2col.par_chunks_mut(k).enumerate().for_each(|(index, row)| {
        let out_x = (index % output_width as usize) as u32;
        let out_y = (index / output_width as usize) as u32;
        let mut idx = 0;
        for ky in 0..kh {
            for kx in 0..kw {
                let pixel_value = image.get_pixel(out_x + kx as u32, out_y + ky as u32)[0] as f32;
                row[idx] = pixel_value;
                idx += 1;
            }
        }
    });
    let mut kernel_flat = Vec::with_capacity(k);
    for ky in 0..kh {
        for kx in 0..kw {
            kernel_flat.push(kernel[ky][kx]);
        }
    }
    let mut output_col = vec![0.0f32; m];
    unsafe {
        sgemm(
            m,
            k,
            n,
            1.0,
            im2col.as_ptr(),
            1,
            k as isize,
            kernel_flat.as_ptr(),
            1,
            0,
            0.0,
            output_col.as_mut_ptr(),
            1,
            0,
        );
    }
    let mut buffer = vec![0u8; m];
    buffer.par_iter_mut().enumerate().for_each(|(i, b)| {
        let val = output_col[i].clamp(0.0, 255.0);
        *b = val as u8;
    });
    let mut output = ImageBuffer::new(output_width, output_height);
    for (i, pixel) in buffer.iter().enumerate() {
        let x = (i % output_width as usize) as u32;
        let y = (i / output_width as usize) as u32;
        output.put_pixel(x, y, Luma([*pixel]));
    }
    output
}