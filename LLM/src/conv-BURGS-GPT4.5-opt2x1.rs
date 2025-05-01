use image::{GrayImage, ImageBuffer, Luma};
use ndarray::{Array2, s, Zip};
use ndarray::linalg::general_mat_mul;

pub fn convolve(image: &GrayImage, kernel: &[&[f32]]) -> GrayImage {
    let (kh, kw) = (kernel.len(), kernel[0].len());
    let kernel = Array2::from_shape_vec((kh, kw), kernel.iter().flat_map(|r| r.iter()).cloned().collect()).unwrap();
    let (width, height) = image.dimensions();
    let output_width = width - kw as u32 + 1;
    let output_height = height - kh as u32 + 1;
    let block_size = 32;
    let img_arr = Array2::from_shape_fn((height as usize, width as usize), |(y, x)| image.get_pixel(x as u32, y as u32)[0] as f32);
    let mut output_arr = Array2::zeros((output_height as usize, output_width as usize));
    let kernel_flat = kernel.as_standard_layout().iter().cloned().collect::<Vec<f32>>();
    for by in (0..output_height as usize).step_by(block_size) {
        for bx in (0..output_width as usize).step_by(block_size) {
            let ymax = (by + block_size).min(output_height as usize);
            let xmax = (bx + block_size).min(output_width as usize);
            let block_height = ymax - by;
            let block_width = xmax - bx;
            let mut block_mat = Array2::zeros((block_height * block_width, kh * kw));
            for y in 0..block_height {
                for x in 0..block_width {
                    let window = img_arr.slice(s![by + y..by + y + kh, bx + x..bx + x + kw]);
                    block_mat.slice_mut(s![y * block_width + x, ..]).assign(&window.iter().cloned().collect::<Array2<f32>>().into_shape(kh * kw).unwrap());
                }
            }
            let mut result = Array2::zeros((block_height * block_width, 1));
            general_mat_mul(1.0, &block_mat, &Array2::from_shape_vec((kh * kw, 1), kernel_flat.clone()).unwrap(), 0.0, &mut result);
            for y in 0..block_height {
                for x in 0..block_width {
                    output_arr[[by + y, bx + x]] = result[[y * block_width + x, 0]].clamp(0.0, 255.0);
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