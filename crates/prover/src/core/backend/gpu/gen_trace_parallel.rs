use std::time::Instant;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

const N_ROWS: u32 = 1 << 5;
const N_STATE: u32 = 16;
const N_INSTANCES_PER_ROW: u32 = 1 << N_LOG_INSTANCES_PER_ROW;
const N_LOG_INSTANCES_PER_ROW: u32 = 3;
const N_COLUMNS: u32 = N_INSTANCES_PER_ROW * N_COLUMNS_PER_REP;
const N_HALF_FULL_ROUNDS: u32 = 4;
const FULL_ROUNDS: u32 = 2 * N_HALF_FULL_ROUNDS;
const N_PARTIAL_ROUNDS: u32 = 14;
const N_LANES: u32 = 16;
const N_COLUMNS_PER_REP: u32 = N_STATE * (1 + FULL_ROUNDS) + N_PARTIAL_ROUNDS;
const LOG_N_LANES: u32 = 4;
const WORKGROUP_SIZE: u32 = 1;
const THREADS_PER_WORKGROUP: u32 = 1 << 5;

use crate::core::backend::simd::column::BaseColumn;
#[allow(unused_imports)]
use crate::core::backend::simd::m31::PackedM31;
#[allow(unused_imports)]
use crate::core::backend::Column;
#[allow(unused_imports)]
use crate::core::fields::m31::M31;
#[allow(unused_imports)]
use crate::examples::poseidon::LookupData;

#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct Complex {
    real: f32,
    imag: f32,
}

#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct GenTraceInput {
    log_n_rows: u32,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct GpuLookupData {
    initial_state: [[GpuBaseColumn; N_STATE as usize]; N_INSTANCES_PER_ROW as usize],
    final_state: [[GpuBaseColumn; N_STATE as usize]; N_INSTANCES_PER_ROW as usize],
}

impl From<GpuLookupData> for LookupData {
    fn from(value: GpuLookupData) -> Self {
        LookupData {
            initial_state: value.initial_state.map(|c| c.map(|c| c.into())),
            final_state: value.final_state.map(|c| c.map(|c| c.into())),
        }
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct GpuPackedM31 {
    data: [u32; N_LANES as usize],
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct StateData {
    data: [GpuPackedM31; N_STATE as usize],
}

#[derive(Clone, Debug, Copy)]
pub struct GpuBaseColumn {
    data: [GpuPackedM31; N_ROWS as usize],
    length: u32,
}

impl GpuBaseColumn {
    fn zeros(length: u32) -> Self {
        GpuBaseColumn {
            data: [GpuPackedM31 {
                data: [0; N_LANES as usize],
            }; N_ROWS as usize],
            length,
        }
    }
}

impl From<GpuBaseColumn> for BaseColumn {
    fn from(value: GpuBaseColumn) -> Self {
        BaseColumn {
            data: value
                .data
                .iter()
                .map(|f| PackedM31::from_array(f.data.map(|v| M31(v))))
                .collect(),
            length: value.length as usize,
        }
    }
}

#[allow(dead_code)]
const GEN_TRACE_OUTPUT_SIZE: usize =
    N_COLUMNS as usize * N_LANES as usize * N_STATE as usize * N_LOG_INSTANCES_PER_ROW as usize;

#[derive(Clone, Debug)]
#[repr(C)]
struct GenTraceOutput {
    trace: [GpuBaseColumn; N_COLUMNS as usize],
    lookup_data: GpuLookupData,
}

impl Default for GenTraceOutput {
    fn default() -> Self {
        GenTraceOutput {
            trace: std::array::from_fn(|_| GpuBaseColumn::zeros(1 << N_LANES)),
            lookup_data: GpuLookupData {
                initial_state: std::array::from_fn(|_| {
                    std::array::from_fn(|_| GpuBaseColumn::zeros(1 << N_LANES))
                }),
                final_state: std::array::from_fn(|_| {
                    std::array::from_fn(|_| GpuBaseColumn::zeros(1 << N_LANES))
                }),
            },
        }
    }
}

// impl GenTraceOutput {
//     fn into_trace(data: &[u8], log_n_rows: u32) -> Vec<BaseColumn> {
//         result.trace.map(|c| c.into_base_column()).to_vec()
//     }
// }

#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct ShaderResult {
    values: [u32; THREADS_PER_WORKGROUP as usize * WORKGROUP_SIZE as usize],
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
        println!("from_bytes");
        println!("bytes.len(): {}", bytes.len());
        println!(
            "std::mem::size_of::<Self>(): {}",
            std::mem::size_of::<Self>()
        );
        assert!(bytes.len() >= std::mem::size_of::<Self>());
        unsafe { &*(bytes.as_ptr() as *const Self) }
    }
}

impl ByteSerialize for GenTraceInput {}
impl ByteSerialize for BaseColumn {}
impl ByteSerialize for StateData {}
impl ByteSerialize for GenTraceOutput {}
impl ByteSerialize for ShaderResult {}
pub async fn gen_trace_parallel(log_n_rows: u32) -> (Vec<BaseColumn>, LookupData) {
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

    let input_data: GenTraceInput = GenTraceInput { log_n_rows };

    // Create buffers
    let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Input Buffer"),
        contents: bytemuck::cast_slice(&[input_data]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let size = std::mem::size_of::<GpuBaseColumn>();
    println!("std::mem::size_of::<GpuBaseColumn>: {}", size);

    println!(
        "std::mem::size_of::<GpuPackedM31>: {}",
        std::mem::size_of::<GpuPackedM31>()
    );
    println!("std::mem::size_of::<u32>: {}", std::mem::size_of::<u32>());
    println!(
        "((std::mem::size_of::<GpuPackedM31>() * N_STATE as usize + std::mem::size_of::<u32>())
            * N_COLUMNS as usize
            * (1 << (log_n_rows - LOG_N_LANES)) as usize): {}",
        ((std::mem::size_of::<GpuPackedM31>() * N_ROWS as usize + std::mem::size_of::<u32>())
            * N_COLUMNS as usize
            * (1 << (log_n_rows - LOG_N_LANES)) as usize)
    );
    println!(
        "std::mem::size_of::<GenTraceOutput>: {}",
        std::mem::size_of::<GenTraceOutput>()
    );
    // let buffer_size = (std::mem::size_of::<GpuPackedM31>() * N_STATE as usize
    //     + std::mem::size_of::<u32>())
    //     * N_COLUMNS as usize
    //     * (1 << (log_n_rows - LOG_N_LANES)) as usize;
    let buffer_size = std::mem::size_of::<GenTraceOutput>();
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Buffer"),
        size: buffer_size as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let state_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("State Buffer"),
        size: (std::mem::size_of::<StateData>()) as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let shader_result = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Shader Result Buffer"),
        size: (std::mem::size_of::<ShaderResult>()) as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Load shader
    let shader_source = include_str!("gen_trace_parallel.wgsl");
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Gen Trace Shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    // Get the maximum buffer size supported by the device
    let max_buffer_size = device.limits().max_buffer_size;
    println!("Maximum buffer size supported: {} bytes", max_buffer_size);

    // Check if our buffer size exceeds the limit
    if buffer_size > max_buffer_size as usize {
        panic!(
            "Required buffer size {} exceeds device maximum of {}",
            buffer_size, max_buffer_size
        );
    }

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
            // Binding 3: Workgroup result buffer
            wgpu::BindGroupLayoutEntry {
                binding: 3,
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
                resource: state_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: shader_result.as_entire_binding(),
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
        compute_pass.dispatch_workgroups(WORKGROUP_SIZE, 1, 1);
    }

    // Copy output to staging buffer for read access
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size: buffer_size as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, staging_buffer.size());

    // create storage buffer for debug data
    let state_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("State Staging Buffer"),
        size: (std::mem::size_of::<StateData>()) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // create storage buffer for workgroup result
    let shader_result_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Shader Result Staging Buffer"),
        size: (std::mem::size_of::<ShaderResult>()) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    encoder.copy_buffer_to_buffer(
        &state_buffer,
        0,
        &state_staging_buffer,
        0,
        state_staging_buffer.size(),
    );

    encoder.copy_buffer_to_buffer(
        &shader_result,
        0,
        &shader_result_staging_buffer,
        0,
        shader_result_staging_buffer.size(),
    );

    // Submit the commands
    queue.submit(Some(encoder.finish()));

    // let buffer_slice = state_staging_buffer.slice(..);
    // let (sender, receiver) = flume::bounded(1);
    // buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
    // device.poll(wgpu::Maintain::wait()).panic_on_timeout();

    // if let Ok(Ok(())) = receiver.recv_async().await {
    //     let data = buffer_slice.get_mapped_range();
    //     let _result = *StateData::from_bytes(&data);
    //     drop(data);
    //     state_staging_buffer.unmap();

    //     println!("State data: {:?}", _result);
    // }

    let buffer_slice = shader_result_staging_buffer.slice(..);
    let (sender, receiver) = flume::bounded(1);
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
    device.poll(wgpu::Maintain::wait()).panic_on_timeout();

    if let Ok(Ok(())) = receiver.recv_async().await {
        let data = buffer_slice.get_mapped_range();
        let _result = *ShaderResult::from_bytes(&data);
        drop(data);
        shader_result_staging_buffer.unmap();

        println!("Shader result: {:?}", _result);
    }

    // Wait for the GPU to finish and map the staging buffer
    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = flume::bounded(1);
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
    device.poll(wgpu::Maintain::wait()).panic_on_timeout();
    println!("after device.poll");
    // let mut result = Vec::<BaseColumn>::new();
    let mut result = GenTraceOutput::default();
    if let Ok(Ok(())) = receiver.recv_async().await {
        println!("inside buffer_slice");
        let data = buffer_slice.get_mapped_range();
        println!("data.len(): {}", data.len());
        result = GenTraceOutput::from_bytes(&data).clone();
        // result = GenTraceOutput::into_trace(&data, log_n_rows);
        drop(data);
        staging_buffer.unmap();
        // println!("result: {:?}", result);

        // let trace = result.trace.clone();
        // for i in 0..trace.len() {
        // println!("trace[{}].data.len(): {:?}", i, trace[i].data.len());
        // }
        // for i in 0..trace.len() {
        //     println!("trace[{}].data[15]: {:?}", i, trace[i].data[15]);
        // }
        // for i in 0..trace[0].data.len() {
        //     println!("trace[0].data[{}]: {:?}", i, trace[0].data[i]);
        // }
        // for i in 0..trace[1].data.len() {
        //     println!("trace[1].data[{}]: {:?}", i, trace[1].data[i]);
        // }
        // for i in 0..trace[2].data.len() {
        //     println!("trace[2].data[{}]: {:?}", i, trace[2].data[i]);
        // }
        // for i in 158..158 * 2 {
        //     println!("trace[{}].data[{}]: {:?}", i, 0, trace[i].data[0]);
        // }
        // let trace = result.trace.clone();
        // for i in 0..trace.len() {
        // println!("trace[{}].data.len(): {:?}", i, trace[i].data.len());
        // }
    }
    println!("Poseidon generate trace time: {:?}", gpu_start.elapsed());

    (
        result.trace.map(|c| c.into()).to_vec(),
        result.lookup_data.into(),
    )
}
