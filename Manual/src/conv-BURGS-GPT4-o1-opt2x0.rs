use once_cell::sync::Lazy;
use std::sync::Mutex;
use image::{GrayImage, ImageBuffer, Luma};
use ocl::{Buffer, Platform, Device, Context, Queue, Program, Kernel};
use std::time::Instant;

static TUNED_LWS: Lazy<Mutex<Option<usize>>> = Lazy::new(|| Mutex::new(None));

pub fn convolve(image: &GrayImage, kernel: &[&[f32]]) -> GrayImage {
    let (kh, kw) = (kernel.len(), kernel[0].len());
    let (width, height) = image.dimensions();
    let out_w = width - kw as u32 + 1;
    let out_h = height - kh as u32 + 1;
    let input_data: Vec<u8> = image.pixels().map(|p| p[0]).collect();
    let kernel_data: Vec<f32> = kernel.iter().flat_map(|row| row.iter()).copied().collect();
    let len_out = (out_w * out_h) as usize;
    let platform = Platform::default();
    let device = Device::first(platform).unwrap();
    let context = Context::builder().devices(device.clone()).build().unwrap();
    let queue = Queue::new(&context, device, None).unwrap();
    let program_src = r#"
__kernel void convolve_kernel(__global uchar *input,__global float *krnl,__global uchar *output,uint width,uint height,uint kw,uint kh,uint outw,uint outh){
   uint x=get_global_id(0);
   uint y=get_global_id(1);
   if(x<outw && y<outh){
      float acc=0.0f;
      for(uint ky=0; ky<kh; ky++){
         for(uint kx=0; kx<kw; kx++){
            float px=(float)input[(y+ky)*width+(x+kx)];
            float w=krnl[ky*kw+kx];
            acc+=px*w;
         }
      }
      if(acc<0.0f)acc=0.0f; 
      if(acc>255.0f)acc=255.0f;
      output[y*outw+x]=(uchar)acc;
   }
}
"#;
    let program = Program::builder().devices(device).src(program_src).build(&context).unwrap();
    let input_buffer = Buffer::<u8>::builder().queue(queue.clone()).len(input_data.len()).build().unwrap();
    let kernel_buffer = Buffer::<f32>::builder().queue(queue.clone()).len(kernel_data.len()).build().unwrap();
    let output_buffer = Buffer::<u8>::builder().queue(queue.clone()).len(len_out).build().unwrap();
    input_buffer.write(&input_data).enq().unwrap();
    kernel_buffer.write(&kernel_data).enq().unwrap();
    let mut best_lws = {
        let mut l = TUNED_LWS.lock().unwrap();
        match *l {
            Some(v) => v,
            None => 0
        }
    };
    if best_lws == 0 {
        let mut best_time = None;
        for &candidate in &[4usize,8,16,32] {
            let k = Kernel::builder()
                .program(&program)
                .name("convolve_kernel")
                .queue(queue.clone())
                .global_work_size([out_w as usize, out_h as usize])
                .local_work_size([candidate, candidate])
                .arg(&input_buffer)
                .arg(&kernel_buffer)
                .arg(&output_buffer)
                .arg(&width)
                .arg(&height)
                .arg(&(kw as u32))
                .arg(&(kh as u32))
                .arg(&out_w)
                .arg(&out_h)
                .build().unwrap();
            let start = Instant::now();
            unsafe { k.enq().unwrap() };
            queue.finish().unwrap();
            let elapsed = start.elapsed().as_nanos();
            if best_time.is_none() || elapsed < best_time.unwrap() {
                best_time = Some(elapsed);
                best_lws = candidate;
            }
        }
        let mut l = TUNED_LWS.lock().unwrap();
        *l = Some(best_lws);
    }
    let k = Kernel::builder()
        .program(&program)
        .name("convolve_kernel")
        .queue(queue.clone())
        .global_work_size([out_w as usize, out_h as usize])
        .local_work_size([best_lws, best_lws])
        .arg(&input_buffer)
        .arg(&kernel_buffer)
        .arg(&output_buffer)
        .arg(&width)
        .arg(&height)
        .arg(&(kw as u32))
        .arg(&(kh as u32))
        .arg(&out_w)
        .arg(&out_h)
        .build().unwrap();
    unsafe { k.enq().unwrap() };
    queue.finish().unwrap();
    let mut result_vec = vec![0u8; len_out];
    output_buffer.read(&mut result_vec).enq().unwrap();
    let mut output = ImageBuffer::<Luma<u8>, Vec<u8>>::new(out_w, out_h);
    for (i, pix) in result_vec.iter().enumerate() {
        let x = (i % out_w as usize) as u32;
        let y = (i / out_w as usize) as u32;
        output.put_pixel(x, y, Luma([*pix]));
    }
    output
}