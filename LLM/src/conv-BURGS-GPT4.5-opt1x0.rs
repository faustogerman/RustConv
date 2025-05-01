use image::{GrayImage, ImageBuffer, Luma};
use ndarray::{Array2, s};
use ndarray::Zip;

pub fn convolve(image: &GrayImage, kernel: &[&[f32]]) -> GrayImage {
    let (kh, kw) = (kernel.len(), kernel[0].len());
    let (width, height) = image.dimensions();
    let output_width = width - kw as u32 + 1;
    let output_height = height - kh as u32 + 1;
    let block_size = 64;
    let input_array: Array2<f32> = Array2::from_shape_fn((height as usize, width as usize), |(y, x)| {
        image.get_pixel(x as u32, y as u32)[0] as f32
    });
    let kernel_array: Array2<f32> = Array2::from_shape_fn((kh, kw), |(y, x)| kernel[y][x]);
    let mut output_array = Array2::<f32>::zeros((output_height as usize, output_width as usize));
    for y_block in (0..output_height as usize).step_by(block_size) {
        for x_block in (0..output_width as usize).step_by(block_size) {
            let y_max = (y_block + block_size).min(output_height as usize);
            let x_max = (x_block + block_size).min(output_width as usize);
            for y in y_block..y_max {
                for x in x_block..x_max {
                    let window = input_array.slice(s![y..y + kh, x..x + kw]);
                    let sum = (&window * &kernel_array).sum();
                    output_array[(y, x)] = sum.clamp(0.0, 255.0);
                }
            }
        }
    }
    let mut output = ImageBuffer::new(output_width, output_height);
    Zip::indexed(&output_array).for_each(|(y, x), &val| {
        output.put_pixel(x as u32, y as u32, Luma([val as u8]));
    });
    output
}