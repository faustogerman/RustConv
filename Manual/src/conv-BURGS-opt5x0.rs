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
    output.enumerate_pixels_mut().for_each(|(x, y, pixel)| {
        let val = conv[(y as usize, x as usize)].clamp(0.0, 255.0) as u8;
        *pixel = Luma([val]);
    });
    output
}