mod conv;
mod filters;
mod utils;

use crate::conv::convolve;
use allocation_counter;
use clap::{Parser, ValueEnum};
use image::ImageReader;
use indicatif::ProgressBar;
use serde_json::json;
use std::cmp::PartialEq;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

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
    filename: PathBuf,
    #[clap(long, value_enum)]
    filter: FiltersKind,
    #[clap(long)]
    size: FilterSizeKind,

    #[arg(long, value_enum, default_value_t=DilationKind::Null)]
    dilation: DilationKind,
}

const NUM_RUNS: u64 = 1;

fn main() {
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
            FilterSizeKind::Large => panic!("Large Gaussian filter not supported."),
        },
    };

    let folder_path = PathBuf::from(&args.filename);
    let entries: Vec<_> = fs::read_dir(folder_path)
        .expect("Invalid directory")
        .filter_map(Result::ok)
        .collect();
    let mut results = vec![];

    let progress = ProgressBar::new(entries.len() as u64);

    for entry in entries {
        let path = entry.path();
        if path.is_file() {
            if let Ok(reader) = ImageReader::open(&path) {
                if let Ok(img) = reader.decode() {
                    let gray_img = img.to_luma8();
                    let mut total_time = 0;
                    let mut total_peak_mem = 0;
                    let mut total_mem = 0;

                    for _ in 0..NUM_RUNS {
                        let start = Instant::now();
                        let info = allocation_counter::measure(|| {
                            let _ = convolve(&gray_img, filter);
                        });
                        let duration = start.elapsed().as_millis();

                        total_time += duration;
                        total_peak_mem += info.bytes_max;
                        total_mem += info.bytes_total;
                    }

                    results.push(json!({
                        "file": path.file_name().unwrap().to_string_lossy(),
                        "memory_peak_bytes": total_peak_mem / NUM_RUNS,
                        "memory_total_bytes": total_mem / NUM_RUNS,
                        "time_seconds": total_time / NUM_RUNS as u128,
                    }));
                }
            }
        }

        progress.inc(1);
    }

    println!("{}", json!(results));
}
