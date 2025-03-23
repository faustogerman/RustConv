use image::{GrayImage, ImageBuffer, Luma};
use ndarray::{Array2, s, Zip};
use wgpu::util::DeviceExt;
use futures::executor::block_on;

async fn gpu_convolve(image: &GrayImage, kernel: &[&[f32]]) -> GrayImage {
    let (width, height) = image.dimensions();
    let (kh, kw) = (kernel.len(), kernel[0].len());
    let output_width = width - kw as u32 + 1;
    let output_height = height - kh as u32 + 1;
    let kernel_flat: Vec<f32> = kernel.iter().flat_map(|r| *r).collect();
    let img_flat: Vec<f32> = image.iter().map(|&p| p as f32).collect();
    let instance = wgpu::Instance::default();
    let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await.unwrap();
    let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor::default(), None).await.unwrap();
    let shader_src = "
        @group(0) @binding(0) var<storage, read> img: array<f32>;
        @group(0) @binding(1) var<storage, read> kernel: array<f32>;
        @group(0) @binding(2) var<storage, read_write> output: array<f32>;
        struct Params { width: u32; height: u32; kw: u32; kh: u32; };
        @group(0) @binding(3) var<uniform> params: Params;
        @compute @workgroup_size(16,16)
        fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
            let x = gid.x;
            let y = gid.y;
            if (x >= params.width || y >= params.height) { return; }
            var acc: f32 = 0.0;
            for (var ky = 0u; ky < params.kh; ky++) {
                for (var kx = 0u; kx < params.kw; kx++) {
                    let ix = x + kx;
                    let iy = y + ky;
                    let img_idx = iy * (params.width + params.kw - 1u) + ix;
                    let k_idx = ky * params.kw + kx;
                    acc += img[img_idx] * kernel[k_idx];
                }
            }
            let out_idx = y * params.width + x;
            output[out_idx] = clamp(acc, 0.0, 255.0);
        }
    ";
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor{
        label: None,
        source: wgpu::ShaderSource::Wgsl(shader_src.into()),
    });
    let img_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
        label: None,
        contents: bytemuck::cast_slice(&img_flat),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let kernel_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
        label: None,
        contents: bytemuck::cast_slice(&kernel_flat),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output_buf = device.create_buffer(&wgpu::BufferDescriptor{
        label: None,
        size: (output_width * output_height * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    #[repr(C)]
    #[derive(Copy, Clone)]
    struct Params{width:u32,height:u32,kw:u32,kh:u32}
    unsafe impl bytemuck::Pod for Params {}
    unsafe impl bytemuck::Zeroable for Params {}
    let params = Params{width:output_width,height:output_height,kw:kw as u32,kh:kh as u32};
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
        label: None,
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor{
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry{binding:0,visibility:wgpu::ShaderStages::COMPUTE,ty:wgpu::BindingType::Buffer{ty:wgpu::BufferBindingType::Storage{read_only:true},has_dynamic_offset:false,min_binding_size:None},count:None},
            wgpu::BindGroupLayoutEntry{binding:1,visibility:wgpu::ShaderStages::COMPUTE,ty:wgpu::BindingType::Buffer{ty:wgpu::BufferBindingType::Storage{read_only:true},has_dynamic_offset:false,min_binding_size:None},count:None},
            wgpu::BindGroupLayoutEntry{binding:2,visibility:wgpu::ShaderStages::COMPUTE,ty:wgpu::BindingType::Buffer{ty:wgpu::BufferBindingType::Storage{read_only:false},has_dynamic_offset:false,min_binding_size:None},count:None},
            wgpu::BindGroupLayoutEntry{binding:3,visibility:wgpu::ShaderStages::COMPUTE,ty:wgpu::BindingType::Buffer{ty:wgpu::BufferBindingType::Uniform,has_dynamic_offset:false,min_binding_size:None},count:None},
        ],
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor{
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry{binding:0,resource:img_buf.as_entire_binding()},
            wgpu::BindGroupEntry{binding:1,resource:kernel_buf.as_entire_binding()},
            wgpu::BindGroupEntry{binding:2,resource:output_buf.as_entire_binding()},
            wgpu::BindGroupEntry{binding:3,resource:params_buf.as_entire_binding()},
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor{
        label: None,
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor{
        label: None,
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor{label:None});
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor{label:None});
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((output_width+15)/16,(output_height+15)/16,1);
    }
    let output_readback = device.create_buffer(&wgpu::BufferDescriptor{
        label: None,
        size: (output_width * output_height * 4) as u64,
        usage: wgpu::BufferUsages::COPY_DST|wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(&output_buf,0,&output_readback,0,(output_width*output_height*4) as u64);
    queue.submit(Some(encoder.finish()));
    let slice = output_readback.slice(..);
    slice.map_async(wgpu::MapMode::Read,|_|());
    device.poll(wgpu::Maintain::Wait);
    let data = slice.get_mapped_range();
    let result: &[f32] = bytemuck::cast_slice(&data);
    let mut output = ImageBuffer::new(output_width, output_height);
    Zip::indexed(&Array2::from_shape_vec((output_height as usize,output_width as usize),result.to_vec()).unwrap()).for_each(|(y,x),&v|{
        output.put_pixel(x as u32,y as u32,Luma([v as u8]));
    });
    output
}