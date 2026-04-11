// Copyright (c) 2026 M.Diach <max@neirosynth.com>
// Proprietary — All Rights Reserved

use metal::*;
use objc::rc::autoreleasepool;

/// Latent Sync Frame Interpolator.
/// Uses Metal to perform motion-vector based interpolation in UMA memory.
pub struct FrameInterpolator {
    device: Device,
    command_queue: CommandQueue,
    pipeline: ComputePipelineState,
}

impl FrameInterpolator {
    pub fn new() -> Option<Self> {
        let device = Device::system_default()?;
        let command_queue = device.new_command_queue();

        // This shader code would typically be in a .metal file.
        // We'll provide a placeholder implementation for the state.
        let library = device
            .new_library_with_source(
                "
            #include <metal_stdlib>
            using namespace metal;
            
            kernel void interpolate(
                texture2d<float, access::read> prev [[texture(0)]],
                texture2d<float, access::read> curr [[texture(1)]],
                texture2d<float, access::write> out [[texture(2)]],
                constant float &alpha [[buffer(3)]],
                uint2 gid [[thread_position_in_grid]])
            {
                if (gid.x >= out.get_width() || gid.y >= out.get_height()) return;
                
                float4 p = prev.read(gid);
                float4 c = curr.read(gid);
                float4 result = mix(p, c, alpha);
                out.write(result, gid);
            }
            ",
                &CompileOptions::new(),
            )
            .ok()?;

        let function = library.get_function("interpolate", None).ok()?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .ok()?;

        Some(Self {
            device,
            command_queue,
            pipeline,
        })
    }

    pub fn interpolate(&self, prev: &Texture, curr: &Texture, output: &Texture, alpha: f32) {
        autoreleasepool(|| {
            let command_buffer = self.command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            encoder.set_compute_pipeline_state(&self.pipeline);
            encoder.set_texture(0, Some(prev));
            encoder.set_texture(1, Some(curr));
            encoder.set_texture(2, Some(output));
            encoder.set_bytes(3, 4, &alpha as *const f32 as *const _);

            // Dispatch threads
            let w = self.pipeline.thread_execution_width();
            let h = self.pipeline.max_total_threads_per_threadgroup() / w;
            let threads_per_group = MTLSize {
                width: w,
                height: h,
                depth: 1,
            };

            let groups = MTLSize {
                width: (output.width() + w - 1) / w,
                height: (output.height() + h - 1) / h,
                depth: 1,
            };

            encoder.dispatch_thread_groups(groups, threads_per_group);
            encoder.end_encoding();

            command_buffer.commit();
            // In a real-time loop, we would not wait synchronously here
            // but use Metal's presentation sync.
            command_buffer.wait_until_completed();
        });
    }
}
