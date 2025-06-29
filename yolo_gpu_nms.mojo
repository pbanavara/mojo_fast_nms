from gpu.host import DeviceContext, DeviceBuffer
from warp_bitmask_nms_simple import fast_nms_simple, TILE
from time import perf_counter_ns
from python import Python
from python import PythonObject

# YOLO detection structure that can be passed from Python
@value
struct Detection:
    var x1: Float32  # left
    var y1: Float32  # top
    var x2: Float32  # right
    var y2: Float32  # bottom
    var confidence: Float32
    var class_id: Int

    fn area(self) -> Float32:
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    fn intersection(self, other: Self) -> Float32:
        var x1 = max(self.x1, other.x1)
        var y1 = max(self.y1, other.y1)
        var x2 = min(self.x2, other.x2)
        var y2 = min(self.y2, other.y2)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        return (x2 - x1) * (y2 - y1)

    fn iou(self, other: Self) -> Float32:
        var inter = self.intersection(other)
        var uni = self.area() + other.area() - inter
        return inter / uni

# Convert Python list of detections to GPU format
fn python_detections_to_gpu_format(detections: PythonObject, ctx: DeviceContext) raises -> (DeviceBuffer[DType.float32], DeviceBuffer[DType.float32]):
    var num_detections = Python.len(detections)
    var boxes_buf = ctx.enqueue_create_buffer[DType.float32](num_detections * 4)
    var scores_buf = ctx.enqueue_create_buffer[DType.float32](num_detections)
    
    with boxes_buf.map_to_host() as boxes_host:
        with scores_buf.map_to_host() as scores_host:
            for i in range(num_detections):
                var det = detections[i]
                # Convert to YOLO format (y1, x1, y2, x2)
                boxes_host[i * 4 + 0] = det.y1
                boxes_host[i * 4 + 1] = det.x1
                boxes_host[i * 4 + 2] = det.y2
                boxes_host[i * 4 + 3] = det.x2
                scores_host[i] = det.confidence
    
    return (boxes_buf, scores_buf)

# Convert GPU results back to Python list
fn gpu_results_to_python_detections(keep_buf: DeviceBuffer[DType.uint8], original_detections: PythonObject) -> PythonObject:
    var kept_detections = Python.list()
    
    with keep_buf.map_to_host() as keep_host:
        for i in range(Python.len(original_detections)):
            if keep_host[i] == 1:
                kept_detections.append(original_detections[i])
    
    return kept_detections

# Main GPU NMS function that can be called from Python
fn gpu_nms_python(detections: PythonObject, iou_threshold: Float32) raises -> PythonObject:
    """
    Run GPU NMS on a list of detections from Python.
    
    Args:
        detections: Python list of Detection objects
        iou_threshold: IoU threshold for suppression
    
    Returns:
        Python list of kept Detection objects
    """
    var ctx = DeviceContext()
    var (boxes_buf, scores_buf) = python_detections_to_gpu_format(detections, ctx)
    var keep_buf = ctx.enqueue_create_buffer[DType.uint8](Python.len(detections))
    
    # Reset keep buffer
    with keep_buf.map_to_host() as keep_host:
        for i in range(Python.len(detections)):
            keep_host[i] = 0
    
    # Run GPU NMS
    ctx.synchronize()
    fast_nms_simple(ctx, boxes_buf, scores_buf, keep_buf, Python.len(detections), iou_threshold)
    ctx.synchronize()
    
    # Convert results back to Python
    return gpu_results_to_python_detections(keep_buf, detections)

# Benchmark function for Python
fn benchmark_gpu_nms_python(detections: PythonObject, iou_threshold: Float32, num_trials: Int) raises -> PythonObject:
    """
    Benchmark GPU NMS performance from Python.
    
    Args:
        detections: Python list of Detection objects
        iou_threshold: IoU threshold for suppression
        num_trials: Number of trials for benchmarking
    
    Returns:
        Python dict with benchmark results
    """
    var ctx = DeviceContext()
    var (boxes_buf, scores_buf) = python_detections_to_gpu_format(detections, ctx)
    var keep_buf = ctx.enqueue_create_buffer[DType.uint8](Python.len(detections))
    
    var total_ns: Float64 = 0.0
    var results = Python.list()
    
    for trial in range(num_trials):
        # Reset keep buffer
        with keep_buf.map_to_host() as keep_host:
            for i in range(Python.len(detections)):
                keep_host[i] = 0
        
        ctx.synchronize()
        var t0 = perf_counter_ns()
        fast_nms_simple(ctx, boxes_buf, scores_buf, keep_buf, Python.len(detections), iou_threshold)
        ctx.synchronize()
        var t1 = perf_counter_ns()
        var ns = Float64(t1 - t0)
        total_ns = total_ns + ns
        
        # Get results for this trial
        var trial_results = gpu_results_to_python_detections(keep_buf, detections)
        results.append(trial_results)
    
    var avg_ms = total_ns / Float64(num_trials) / 1_000_000.0
    
    # Create result dict
    var result_dict = Python.dict()
    result_dict["average_time_ms"] = avg_ms
    result_dict["total_time_ns"] = total_ns
    result_dict["num_trials"] = num_trials
    result_dict["num_detections"] = Python.len(detections)
    result_dict["final_result"] = results[num_trials - 1]  # Last trial result
    
    return result_dict

# Create Python module
@export
fn PyInit_yolo_gpu_nms() -> PythonObject:
    try:
        var m = PythonModuleBuilder("yolo_gpu_nms")
        
        # Expose the main GPU NMS function
        m.def_function[gpu_nms_python]("gpu_nms", docstring="Run GPU NMS on detections")
        
        # Expose the benchmark function
        m.def_function[benchmark_gpu_nms_python]("benchmark_gpu_nms", docstring="Benchmark GPU NMS performance")
        
        # Expose the Detection struct
        m.def_struct[Detection]("Detection", docstring="YOLO detection structure")
        
        return m.finalize()
    except e:
        return abort[PythonObject](String("error creating Python Mojo module:", e))

# Test function for standalone Mojo execution
def main():
    print("YOLO GPU NMS Module - Ready for Python import!")
    print("Use: import yolo_gpu_nms")
    print("Then: yolo_gpu_nms.gpu_nms(detections, iou_threshold)") 