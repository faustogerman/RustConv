mod filters;

use image::{GrayImage, ImageBuffer, Luma};
use image::{ImageReader, RgbImage};
use ndarray::{ArrayView, Dim, Ix};

fn img_to_array(img: &RgbImage) -> ArrayView<u8, Dim<[Ix; 3]>> {
    // We could use 16 or 32 bits. Not sure if >8 bits makes any difference.

    let raw_pixel_data = img.as_raw(); // Vec[u8] of length h*w
    let (w, h) = img.dimensions();

    ArrayView::from_shape((h as usize, w as usize, 3), raw_pixel_data)
        .expect("Shape mismatch with raw image data.")
}

fn main() {
    let img = ImageReader::open("../Images/collection/1d8ef901.jpg")
        .expect("Image could not be loaded")
        .decode()
        .expect("Image could not be decoded")
        .to_luma8();

    let result = convolve(&img, &filters::BOX_BLUR_3X3);
    result
        .save("./_out_images_test/output1.jpg")
        .expect("Failed to save image");
    let result = convolve(&img, &filters::GAUSSIAN_6X6);
    result
        .save("./_out_images_test/output2.jpg")
        .expect("Failed to save image");
    let result = convolve(&img, &filters::RIDGE_12X12);
    result
        .save("./_out_images_test/output3.jpg")
        .expect("Failed to save image");
}

/// Applies a convolution operation on a grayscale image using a kernel of any numeric type and size.
pub fn convolve<const H: usize, const W: usize>(
    image: &GrayImage,
    kernel: &[[f32; W]; H],
) -> GrayImage {
    let (width, height) = image.dimensions();
    let output_width = width - W as u32 + 1;
    let output_height = height - H as u32 + 1;

    let mut output = ImageBuffer::new(output_width, output_height);

    for y in 0..output_height {
        for x in 0..output_width {
            let mut acc = 0.0;

            for ky in 0..H {
                for kx in 0..W {
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
