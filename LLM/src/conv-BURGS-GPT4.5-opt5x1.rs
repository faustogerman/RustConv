use image::{GrayImage, ImageBuffer, Luma};
use ndarray::{Array2, s};
use ndarray_conv::Conv2D;
use ndarray::Zip;

pub fn convolve(image: &GrayImage, kernel: &[&[f32]]) -> GrayImage {
    let block_size = 64;
    let (width, height) = image.dimensions();
    let img_array = Array2::from_shape_fn((height as usize, width as usize), |(y, x)| {
        image.get_pixel(x as u32, y as u32)[0] as f32
    });
    let kernel_array = Array2::from_shape_fn((kernel.len(), kernel[0].len()), |(y, x)| kernel[y][x]);
    let (k_height, k_width) = kernel_array.dim();
    let conv_height = height as usize - k_height + 1;
    let conv_width = width as usize - k_width + 1;
    let mut output_array = Array2::<f32>::zeros((conv_height, conv_width));
    for y_block in (0..conv_height).step_by(block_size) {
        for x_block in (0..conv_width).step_by(block_size) {
            let y_max = (y_block + block_size).min(conv_height);
            let x_max = (x_block + block_size).min(conv_width);
            for y in y_block..y_max {
                for x in x_block..x_max {
                    let window = img_array.slice(s![y..y + k_height, x..x + k_width]);
                    let mut sum = 0.0;
                    Zip::from(&window).and(&kernel_array).for_each(|a, &b| sum += a * b);
                    output_array[[y, x]] = sum;
                }
            }
        }
    }
    let mut output = ImageBuffer::new(conv_width as u32, conv_height as u32);
    output.enumerate_pixels_mut().for_each(|(x, y, pixel)| {
        let val = output_array[(y as usize, x as usize)].clamp(0.0, 255.0) as u8;
        *pixel = Luma([val]);
    });
    output
}