use std::time::Instant;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

// use crate::constraint_framework::EvalAtRow;
// use crate::examples::poseidon::PoseidonElements;

const N_STATE: u32 = 16;
#[allow(dead_code)]
const N_INSTANCES_PER_ROW: u32 = 8;
#[allow(dead_code)]
const N_COLUMNS: u32 = N_INSTANCES_PER_ROW * N_COLUMNS_PER_REP;
#[allow(dead_code)]
const N_HALF_FULL_ROUNDS: u32 = 4;
#[allow(dead_code)]
const FULL_ROUNDS: u32 = 2 * N_HALF_FULL_ROUNDS;
#[allow(dead_code)]
const N_PARTIAL_ROUNDS: u32 = 14;
const N_LANES: u32 = 16;
#[allow(dead_code)]
const N_COLUMNS_PER_REP: u32 = N_STATE * (1 + FULL_ROUNDS) + N_PARTIAL_ROUNDS;
#[allow(dead_code)]
const LOG_N_LANES: u32 = 4;

#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct Complex {
    real: f32,
    imag: f32,
}

#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct GenTraceInput {
    log_size: u32,
}

#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct BaseColumn {
    data: [PackedM31; N_STATE as usize],
    length: u32,
}

#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct PackedM31 {
    data: [u32; N_LANES as usize],
}

#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct LookupData {
    initial_state: [[BaseColumn; N_STATE as usize]; N_INSTANCES_PER_ROW as usize],
    final_state: [[BaseColumn; N_STATE as usize]; N_INSTANCES_PER_ROW as usize],
}

#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct DebugData {
    index: [u32; 16],
    values: [u32; 16],
    counter: u32,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct GenTraceOutput {
    data: [PackedM31; N_STATE as usize],
    trace: [BaseColumn; N_COLUMNS as usize],
    lookup_data: LookupData,
}

pub trait ByteSerialize: Sized {
    fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                (self as *const Self) as *const u8,
                std::mem::size_of::<Self>(),
            )
        }
    }

    fn from_bytes(bytes: &[u8]) -> &Self {
        assert!(bytes.len() >= std::mem::size_of::<Self>());
        unsafe { &*(bytes.as_ptr() as *const Self) }
    }
}

impl ByteSerialize for GenTraceInput {}
impl ByteSerialize for BaseColumn {}
impl ByteSerialize for DebugData {}
impl ByteSerialize for GenTraceOutput {}

pub async fn gen_trace() {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .unwrap();

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Device"),
                required_features: wgpu::Features::SHADER_INT64,
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        )
        .await
        .unwrap();

    let input_data: GenTraceInput = GenTraceInput { log_size: 7 };

    // Create buffers
    let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Input Buffer"),
        contents: bytemuck::cast_slice(&[input_data]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Buffer"),
        size: (N_STATE as usize * std::mem::size_of::<GenTraceOutput>()) as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let debug_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Debug Buffer"),
        size: (std::mem::size_of::<DebugData>()) as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Load shader
    let shader_source = include_str!("gen_trace.wgsl");
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Gen Trace Shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    // Bind group layout
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        entries: &[
            // Binding 0: Input buffer
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Binding 1: Output buffer
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Binding 2: Debug buffer
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
        label: Some("Gen Trace Bind Group Layout"),
    });

    // Create bind group
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: debug_buffer.as_entire_binding(),
            },
        ],
        label: Some("Gen Trace Bind Group"),
    });

    // Pipeline layout
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
        label: Some("Gen Trace Pipeline Layout"),
    });

    // Compute pipeline
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Gen Trace Compute Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some("gen_trace"),
        cache: None,
        compilation_options: Default::default(),
    });

    // Create encoder
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Gen Trace Command Encoder"),
    });

    // === GPU FFT Timing Start ===
    let gpu_start = Instant::now();

    // Dispatch the compute shader
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Gen Trace Compute Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        // Workgroup size defined in shader
        let workgroup_size = 256u32;

        compute_pass.dispatch_workgroups(workgroup_size, 1, 1);
    }

    // Copy output to staging buffer for read access
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size: (N_STATE as usize * std::mem::size_of::<GenTraceOutput>()) as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, staging_buffer.size());

    // create storage buffer for debug data
    let debug_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Debug Staging Buffer"),
        size: (std::mem::size_of::<DebugData>()) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    encoder.copy_buffer_to_buffer(
        &debug_buffer,
        0,
        &debug_staging_buffer,
        0,
        debug_staging_buffer.size(),
    );

    // Submit the commands
    queue.submit(Some(encoder.finish()));

    {
        let buffer_slice = debug_staging_buffer.slice(..);
        let (sender, receiver) = flume::bounded(1);
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
        device.poll(wgpu::Maintain::wait()).panic_on_timeout();

        if let Ok(Ok(())) = receiver.recv_async().await {
            let data = buffer_slice.get_mapped_range();
            let result = *DebugData::from_bytes(&data);
            drop(data);
            debug_staging_buffer.unmap();

            println!("Debug data: {:?}", result);
        }
    }

    // Wait for the GPU to finish and map the staging buffer
    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = flume::bounded(1);
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
    device.poll(wgpu::Maintain::wait()).panic_on_timeout();

    if let Ok(Ok(())) = receiver.recv_async().await {
        let data = buffer_slice.get_mapped_range();
        let result = *GenTraceOutput::from_bytes(&data);

        drop(data);
        staging_buffer.unmap();

        println!("Output: {:?}", result);
    }

    println!("Poseidon generate trace time: {:?}", gpu_start.elapsed());
}
