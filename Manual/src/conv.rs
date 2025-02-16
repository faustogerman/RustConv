use image::{GrayImage, ImageBuffer, Luma};

/// Applies a convolution operation on a grayscale image using a kernel of any numeric type and size.
pub fn convolve(image: &GrayImage, kernel: &[&[f32]]) -> GrayImage {
    let (kh, kw) = (kernel.len(), kernel[0].len());

    let (width, height) = image.dimensions();
    let output_width = width - kw as u32 + 1;
    let output_height = height - kh as u32 + 1;

    let mut output = ImageBuffer::new(output_width, output_height);

    for y in 0..output_height {
        for x in 0..output_width {
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

    output
}
