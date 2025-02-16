mod conv;
mod filters;
mod utils;

use crate::conv::convolve;
use clap::{Parser, ValueEnum};
use image::ImageReader;
use std::cmp::PartialEq;

#[derive(ValueEnum, Clone, Debug, PartialEq)]
enum DilationKind {
    Null,
    Small,
    Medium,
    Large,
}

#[derive(ValueEnum, Clone, Debug)]
enum FiltersKind {
    Ridge,
    Sharpen,
    BoxBlur,
    Gaussian,
}

#[derive(ValueEnum, Clone, Debug)]
enum FilterSizeKind {
    Small,
    Medium,
    Large,
}

/// Search for a pattern in a file and display the lines that contain it.
#[derive(Parser, Debug)]
struct Cli {
    filename: std::path::PathBuf,
    #[clap(long, value_enum)]
    filter: FiltersKind,
    #[clap(long)]
    size: FilterSizeKind,

    #[arg(long, value_enum, default_value_t=DilationKind::Null)]
    dilation: DilationKind,
}

fn main() {
    // structure: RustConv <filename> <filter> <filter size> <(optional) --dilation>
    // For example (with binary executable): ```RustConv ./path/to/image.jpg --filter RIDGE --size 3
    // For example (with cargo run): ```RustConv -- ./path/to/image.jpg --filter RIDGE --size 3
    // For example (with binary executable): ```RustConv ./path/to/image.jpg --filter RIDGE --size 3 --dilation small
    let args = Cli::parse();

    let filter: &[&[f32]] = match args.filter {
        FiltersKind::Ridge => match args.size {
            FilterSizeKind::Small => filters::RIDGE_3X3,
            FilterSizeKind::Medium => filters::RIDGE_6X6,
            FilterSizeKind::Large => filters::RIDGE_12X12,
        },
        FiltersKind::Sharpen => match args.size {
            FilterSizeKind::Small => filters::SHARPEN_3X3,
            FilterSizeKind::Medium => filters::SHARPEN_6X6,
            FilterSizeKind::Large => filters::SHARPEN_12X12,
        },
        FiltersKind::BoxBlur => match args.size {
            FilterSizeKind::Small => filters::BOX_BLUR_3X3,
            FilterSizeKind::Medium => filters::BOX_BLUR_6X6,
            FilterSizeKind::Large => filters::BOX_BLUR_12X12,
        },
        FiltersKind::Gaussian => match args.size {
            FilterSizeKind::Small => filters::GAUSSIAN_3X3,
            FilterSizeKind::Medium => filters::GAUSSIAN_6X6,
            FilterSizeKind::Large => unimplemented!("Large Gaussian filter is not yet supported."),
        },
    };

    if args.dilation != DilationKind::Null {
        unimplemented!("Dilation is not supported yet");
    }

    let img = ImageReader::open(args.filename)
        .expect("Image could not be loaded")
        .decode()
        .expect("Image could not be decoded")
        .to_luma8();

    convolve(&img, filter);
}
