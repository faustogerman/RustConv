use image::{GrayImage, ImageBuffer, Luma};
use ndarray::{Array2, s, Zip};

pub fn convolve(image: &GrayImage, kernel: &[&[f32]]) -> GrayImage {
    let (kh, kw) = (kernel.len(), kernel[0].len());
    let kernel = Array2::from_shape_vec((kh, kw), kernel.iter().flat_map(|r| r.iter()).cloned().collect()).unwrap();
    let (width, height) = image.dimensions();
    let output_width = width - kw as u32 + 1;
    let output_height = height - kh as u32 + 1;
    let block_size = 32;
    let img_arr = Array2::from_shape_fn((height as usize, width as usize), |(y, x)| image.get_pixel(x as u32, y as u32)[0] as f32);
    let mut output_arr = Array2::zeros((output_height as usize, output_width as usize));
    for by in (0..output_height as usize).step_by(block_size) {
        for bx in (0..output_width as usize).step_by(block_size) {
            let ymax = (by + block_size).min(output_height as usize);
            let xmax = (bx + block_size).min(output_width as usize);
            for y in by..ymax {
                for x in bx..xmax {
                    let window = img_arr.slice(s![y..y + kh, x..x + kw]);
                    let acc = (&window * &kernel).sum();
                    output_arr[[y, x]] = acc.clamp(0.0, 255.0);
                }
            }
        }
    }
    let mut output = ImageBuffer::new(output_width, output_height);
    Zip::indexed(&output_arr).for_each(|(y, x), &val| {
        output.put_pixel(x as u32, y as u32, Luma([val as u8]));
    });
    output
}