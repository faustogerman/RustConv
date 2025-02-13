mod filters;

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
    let img = ImageReader::open("../Images/collection/1d8ef901.png")
        .expect("Image could not be loaded")
        .decode()
        .expect("Image could not be decoded")
        .to_rgb8();

    let xy = img_to_array(&img);
    // println!("{}", xy);
}
