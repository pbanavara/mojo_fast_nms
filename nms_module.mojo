from python import Python, PythonObject
from gpu.host import DeviceContext, DeviceBuffer
from time import perf_counter_ns

# Import our NMS implementation
from warp_bitmask_nms_simple import fast_nms_simple

@export
fn PyInit_nms_module() -> PythonObject:
    try:
        var m = PythonModuleBuilder("nms_module")
        m.def_function[nms_gpu_wrapper]("nms_gpu", docstring="GPU NMS using Mojo")
        m.def_function[benchmark_nms]("benchmark_nms", docstring="Benchmark NMS performance")
        return m.finalize()
    except e:
        return abort[PythonObject](String("error creating Python Mojo module:", e))

fn nms_gpu_wrapper(boxes: PythonObject, scores: PythonObject, iou_threshold: PythonObject) raises -> PythonObject:
    """Wrapper function to call GPU NMS from Python"""
    # Convert Python objects to Mojo types
    var boxes_list = boxes.to_list()
    var scores_list = scores.to_list()
    var iou_th = Float32(iou_threshold.to_float())
    
    var n = boxes_list.len() // 4  # Each box has 4 coordinates
    
    # Create GPU context
    var ctx = DeviceContext()
    
    # Allocate GPU buffers
    var boxes_buf = ctx.enqueue_create_buffer[DType.float32](n * 4)
    var scores_buf = ctx.enqueue_create_buffer[DType.float32](n)
    var keep_buf = ctx.enqueue_create_buffer[DType.uint8](n)
    
    # Copy data to GPU
    with boxes_buf.map_to_host() as boxes_host:
        for i in range(n * 4):
            boxes_host[i] = Float32(boxes_list[i].to_float())
    
    with scores_buf.map_to_host() as scores_host:
        for i in range(n):
            scores_host[i] = Float32(scores_list[i].to_float())
    
    # Run NMS
    fast_nms_simple(ctx, boxes_buf, scores_buf, keep_buf, n, iou_th)
    
    # Get results back
    var keep_indices = DynamicVector[Int]()
    with keep_buf.map_to_host() as keep_host:
        for i in range(n):
            if keep_host[i] != 0:
                keep_indices.push_back(i)
    
    # Convert back to Python list
    var result = Python.list()
    for i in range(keep_indices.len()):
        result.append(keep_indices[i])
    
    return result

fn benchmark_nms(n_boxes: PythonObject, iou_threshold: PythonObject, num_trials: PythonObject) raises -> PythonObject:
    """Benchmark NMS performance"""
    var n = Int(n_boxes.to_int())
    var iou_th = Float32(iou_threshold.to_float())
    var trials = Int(num_trials.to_int())
    
    # Create GPU context
    var ctx = DeviceContext()
    
    # Allocate buffers
    var boxes_buf = ctx.enqueue_create_buffer[DType.float32](n * 4)
    var scores_buf = ctx.enqueue_create_buffer[DType.float32](n)
    var keep_buf = ctx.enqueue_create_buffer[DType.uint8](n)
    
    # Generate random data
    var seed = 42
    with boxes_buf.map_to_host() as boxes_host:
        for i in range(n * 4):
            var rand_val, new_seed = lcg_rand(seed)
            boxes_host[i] = rand_val
            seed = new_seed
    
    with scores_buf.map_to_host() as scores_host:
        for i in range(n):
            var rand_val, new_seed = lcg_rand(seed)
            scores_host[i] = rand_val
            seed = new_seed
    
    # Warm up
    fast_nms_simple(ctx, boxes_buf, scores_buf, keep_buf, n, iou_th)
    
    # Benchmark
    var start_time = perf_counter_ns()
    for _ in range(trials):
        fast_nms_simple(ctx, boxes_buf, scores_buf, keep_buf, n, iou_th)
    var end_time = perf_counter_ns()
    
    var avg_time = Float64(end_time - start_time) / Float64(trials) / 1_000_000.0  # Convert to ms
    
    # Return results as Python dict
    var result = Python.dict()
    result.set_item("avg_time_ms", avg_time)
    result.set_item("num_boxes", n)
    result.set_item("num_trials", trials)
    
    return result

def lcg_rand(seed: Int) -> (Float32, Int):
    """Simple linear congruential generator for random data"""
    var new_seed = (seed * 1103515245 + 12345) & 0x7fffffff
    var rand_val = Float32(new_seed) / Float32(0x7fffffff)
    return (rand_val, new_seed) 