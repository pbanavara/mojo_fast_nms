from gpu.host import DeviceContext, DeviceBuffer
from warp_bitmask_nms_simple import fast_nms_simple, TILE
from time import perf_counter_ns

def lcg_rand(seed: Int) -> (Float32, Int):
    var a = 1664525
    var c = 1013904223
    var m = 2 ** 32
    var next_seed = (a * seed + c) % m
    var rand_float = Float32(next_seed) / Float32(m)
    return (rand_float, next_seed)

def main():
    try:
        print("Testing simplified warp_bitmask_nms module...")
        print("Module imported successfully!")
        print("TILE size:", TILE)
        
        n = 4096
        num_trials = 10
        
        var ctx = DeviceContext()
        var boxes_buf = ctx.enqueue_create_buffer[DType.float32](n * 4)
        var scores_buf = ctx.enqueue_create_buffer[DType.float32](n)
        var keep_buf = ctx.enqueue_create_buffer[DType.uint8](n)
        
        # Generate random test data using LCG
        var seed = 42
        with boxes_buf.map_to_host() as boxes_host:
            for i in range(n):
                var y1, seed1 = lcg_rand(seed)
                var x1, seed2 = lcg_rand(seed1)
                var h_raw, seed3 = lcg_rand(seed2)
                var w_raw, seed4 = lcg_rand(seed3)
                seed = seed4
                var h: Float32 = h_raw * 0.1 + 0.05
                var w: Float32 = w_raw * 0.1 + 0.05
                var y2: Float32 = y1 + h
                var x2: Float32 = x1 + w
                boxes_host[i*4+0] = y1
                boxes_host[i*4+1] = x1
                boxes_host[i*4+2] = y2
                boxes_host[i*4+3] = x2
        with scores_buf.map_to_host() as scores_host:
            for i in range(n):
                var s, seed1 = lcg_rand(seed)
                seed = seed1
                scores_host[i] = s
        
        print("Running NMS with n=", n, ", trials=", num_trials)
        var total_ns: Float64 = 0.0
        for trial in range(num_trials):
            with keep_buf.map_to_host() as keep_host:
                for i in range(n):
                    keep_host[i] = 0
            ctx.synchronize()
            var t0 = perf_counter_ns()
            fast_nms_simple(ctx, boxes_buf, scores_buf, keep_buf, n, 0.5)
            ctx.synchronize()
            var t1 = perf_counter_ns()
            var ns = Float64(t1 - t0)
            total_ns = total_ns + ns
            print("Trial", trial + 1, ":", ns / 1_000_000.0, "ms")
        
        var avg_ms = total_ns / Float64(num_trials) / 1_000_000.0
        print("Mojo NMS (n=", n, ", trials=", num_trials, "): Avg time = ", avg_ms, " ms")
        print("Test completed!")
    except e:
        print("Error:", e)
    finally:
        print("Done") 