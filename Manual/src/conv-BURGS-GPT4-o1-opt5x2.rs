use image::{GrayImage, ImageBuffer, Luma};
use rayon::prelude::*;
use ocl::{ProQue, Buffer};
use std::time::Instant;

pub fn convolve(image: &GrayImage, kernel: &[&[f32]]) -> GrayImage {
    let (kh, kw) = (kernel.len(), kernel[0].len());
    let (width, height) = image.dimensions();
    let output_width = width - kw as u32 + 1;
    let output_height = height - kh as u32 + 1;
    let input_data = image.as_raw();
    let mut kernel_data = Vec::with_capacity(kh * kw);
    for row in kernel {
        kernel_data.extend_from_slice(row);
    }
    let mut output_data = vec![0u8; (output_width * output_height) as usize];
    let src = r#"
__kernel void convolve_kernel(__global uchar* input,__global uchar* output,__global float* k,uint w,uint h,uint kw,uint kh,uint ow,uint oh){int x=get_global_id(0);int y=get_global_id(1);if(x<ow&&y<oh){float acc=0.0f;for(int ky=0;ky<kh;ky++){for(int kx=0;kx<kw;kx++){acc+=((float)input[(y+ky)*w+(x+kx)])*k[ky*kw+kx];}}acc=fmax(0.0f,fmin(255.0f,acc));output[y*ow+x]=(uchar)acc;}}"#;
    let proque = ProQue::builder().src(src).dims((output_width, output_height)).build().unwrap();
    let buffer_input = proque.create_buffer::<u8>().unwrap();
    let buffer_output = proque.create_buffer::<u8>().unwrap();
    let buffer_kernel = proque.create_buffer::<f32>().unwrap();
    buffer_input.write(&input_data).enq().unwrap();
    buffer_kernel.write(&kernel_data).enq().unwrap();
    let mut best_time = std::time::Duration::MAX;
    let mut best_lws = 1;
    for ws in [8,16,32] {
        let k = proque.kernel_builder("convolve_kernel")
            .arg(&buffer_input)
            .arg(&buffer_output)
            .arg(&buffer_kernel)
            .arg(width)
            .arg(height)
            .arg(kw as u32)
            .arg(kh as u32)
            .arg(output_width)
            .arg(output_height)
            .build().unwrap();
        let start = Instant::now();
        unsafe {
            k.cmd()
                .global_work_size((output_width, output_height))
                .local_work_size((ws, ws))
                .enq().unwrap();
        }
        proque.queue().finish().unwrap();
        let dur = start.elapsed();
        if dur < best_time {
            best_time = dur;
            best_lws = ws;
        }
    }
    let k = proque.kernel_builder("convolve_kernel")
        .arg(&buffer_input)
        .arg(&buffer_output)
        .arg(&buffer_kernel)
        .arg(width)
        .arg(height)
        .arg(kw as u32)
        .arg(kh as u32)
        .arg(output_width)
        .arg(output_height)
        .build().unwrap();
    unsafe {
        k.cmd()
            .global_work_size((output_width, output_height))
            .local_work_size((best_lws, best_lws))
            .enq().unwrap();
    }
    proque.queue().finish().unwrap();
    buffer_output.read(&mut output_data).enq().unwrap();
    let mut output = ImageBuffer::new(output_width, output_height);
    for (i, &pix) in output_data.iter().enumerate() {
        let x = (i % output_width as usize) as u32;
        let y = (i / output_width as usize) as u32;
        output.put_pixel(x, y, Luma([pix]));
    }
    output
}