use image::{GrayImage, ImageBuffer, Luma};
use ndarray::{s, Array2};
use ndarray_conv::Conv2D;

pub fn convolve(image: &GrayImage, kernel: &[&[f32]]) -> GrayImage {
    let (width, height) = image.dimensions();
    let img_array = Array2::from_shape_fn((height as usize, width as usize), |(y, x)| {
        image.get_pixel(x as u32, y as u32)[0] as f32
    });
    let kernel_array = Array2::from_shape_fn((kernel.len(), kernel[0].len()), |(y, x)| kernel[y][x]);
    let conv = img_array.conv2d(&kernel_array, Conv2D::Valid);
    let (conv_height, conv_width) = conv.dim();
    let mut output = ImageBuffer::new(conv_width as u32, conv_height as u32);
    let pixels = output.as_mut();
    let mut i = 0;
    let len = pixels.len();
    let conv_flat = conv.as_slice().unwrap();
    let unroll_factor = 4;
    while i + unroll_factor <= len {
        pixels[i] = conv_flat[i].clamp(0.0, 255.0) as u8;
        pixels[i + 1] = conv_flat[i + 1].clamp(0.0, 255.0) as u8;
        pixels[i + 2] = conv_flat[i + 2].clamp(0.0, 255.0) as u8;
        pixels[i + 3] = conv_flat[i + 3].clamp(0.0, 255.0) as u8;
        i += unroll_factor;
    }
    while i < len {
        pixels[i] = conv_flat[i].clamp(0.0, 255.0) as u8;
        i += 1;
    }
    output
}