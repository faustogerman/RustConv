use image::{GrayImage, ImageBuffer, Luma};

pub fn convolve(image: &GrayImage, kernel: &[&[f32]]) -> GrayImage {
    let (kh, kw) = (kernel.len(), kernel[0].len());
    let (width, height) = image.dimensions();
    let output_width = width - kw as u32 + 1;
    let output_height = height - kh as u32 + 1;
    let mut output = ImageBuffer::new(output_width, output_height);
    let block_size = 16u32;
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
                            let pixel_value = image.get_pixel(x + kx as u32, y + ky as u32)[0] as f32;
                            let weight = kernel[ky][kx];
                            acc += pixel_value * weight;
                        }
                    }
                    output.put_pixel(x, y, Luma([acc.clamp(0.0, 255.0) as u8]));
                }
            }
        }
    }

    output
}