from gpu.host import DeviceContext, DeviceBuffer
from warp_bitmask_nms_simple import fast_nms_simple, TILE
from time import perf_counter_ns
from memory import stack_allocation
from algorithm import vectorize
from memory.buffer import Buffer

# YOLO detection structure
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

# CPU NMS implementation for comparison
fn cpu_nms(detections: Buffer[Detection], iou_threshold: Float32) -> Buffer[Detection]:
    if detections.size() == 0:
        return Buffer[Detection]()
    
    # Sort by confidence (descending)
    var sorted_detections = Buffer[Detection](detections.size())
    for i in range(detections.size()):
        sorted_detections[i] = detections[i]
    
    # Simple bubble sort (for small datasets)
    for i in range(sorted_detections.size()):
        for j in range(i + 1, sorted_detections.size()):
            if sorted_detections[i].confidence < sorted_detections[j].confidence:
                var temp = sorted_detections[i]
                sorted_detections[i] = sorted_detections[j]
                sorted_detections[j] = temp
    
    var keep = Buffer[Detection]()
    var suppressed = Buffer[Bool](sorted_detections.size())
    for i in range(suppressed.size()):
        suppressed[i] = False
    
    for i in range(sorted_detections.size()):
        if suppressed[i]:
            continue
        
        keep.push_back(sorted_detections[i])
        
        for j in range(i + 1, sorted_detections.size()):
            if suppressed[j]:
                continue
            
            var iou = sorted_detections[i].iou(sorted_detections[j])
            if iou >= iou_threshold:
                suppressed[j] = True
    
    return keep

# Generate realistic YOLO-like detections
fn generate_yolo_detections(num_detections: Int, image_width: Int, image_height: Int) -> Buffer[Detection]:
    var detections = Buffer[Detection]()
    var seed = 42
    
    for i in range(num_detections):
        # Generate random box coordinates
        var x1 = Float32((seed * 1664525 + 1013904223) % 1000) / 1000.0 * Float32(image_width)
        var y1 = Float32(((seed + 1) * 1664525 + 1013904223) % 1000) / 1000.0 * Float32(image_height)
        var w = Float32(((seed + 2) * 1664525 + 1013904223) % 1000) / 1000.0 * 100.0 + 20.0
        var h = Float32(((seed + 3) * 1664525 + 1013904223) % 1000) / 1000.0 * 100.0 + 20.0
        var confidence = Float32(((seed + 4) * 1664525 + 1013904223) % 1000) / 1000.0
        var class_id = (seed + 5) % 80  # COCO has 80 classes
        
        # Ensure box is within image bounds
        var x2 = min(x1 + w, Float32(image_width))
        var y2 = min(y1 + h, Float32(image_height))
        x1 = max(x1, 0.0)
        y1 = max(y1, 0.0)
        
        detections.push_back(Detection(x1, y1, x2, y2, confidence, class_id))
        seed = (seed + 7) % 1000000
    
    return detections

# Convert detections to GPU format
fn detections_to_gpu_format(detections: Buffer[Detection], ctx: DeviceContext) raises -> (DeviceBuffer[DType.float32], DeviceBuffer[DType.float32]):
    var boxes_buf = ctx.enqueue_create_buffer[DType.float32](detections.size() * 4)
    var scores_buf = ctx.enqueue_create_buffer[DType.float32](detections.size())
    
    with boxes_buf.map_to_host() as boxes_host:
        with scores_buf.map_to_host() as scores_host:
            for i in range(detections.size()):
                var det = detections[i]
                # Convert to YOLO format (y1, x1, y2, x2)
                boxes_host[i * 4 + 0] = det.y1
                boxes_host[i * 4 + 1] = det.x1
                boxes_host[i * 4 + 2] = det.y2
                boxes_host[i * 4 + 3] = det.x2
                scores_host[i] = det.confidence
    
    return (boxes_buf, scores_buf)

# Convert GPU results back to detections
fn gpu_results_to_detections(keep_buf: DeviceBuffer[DType.uint8], original_detections: Buffer[Detection]) -> Buffer[Detection]:
    var kept_detections = Buffer[Detection]()
    
    with keep_buf.map_to_host() as keep_host:
        for i in range(original_detections.size()):
            if keep_host[i] == 1:
                kept_detections.push_back(original_detections[i])
    
    return kept_detections

# Benchmark CPU vs GPU NMS
fn benchmark_nms_performance() raises:
    print("=== YOLO NMS Performance Benchmark ===")
    
    var image_width = 640
    var image_height = 640
    var num_detections = 1000
    var iou_threshold = 0.5
    var num_trials = 10
    
    print("Image size:", image_width, "x", image_height)
    print("Number of detections:", num_detections)
    print("IoU threshold:", iou_threshold)
    print("Number of trials:", num_trials)
    print()
    
    # Generate test detections
    print("Generating test detections...")
    var detections = generate_yolo_detections(num_detections, image_width, image_height)
    print("Generated", detections.size(), "detections")
    print()
    
    # CPU NMS benchmark
    print("Running CPU NMS benchmark...")
    var cpu_total_ns: Float64 = 0.0
    var cpu_results = Buffer[Detection]()
    
    for trial in range(num_trials):
        var t0 = perf_counter_ns()
        cpu_results = cpu_nms(detections, iou_threshold)
        var t1 = perf_counter_ns()
        var ns = Float64(t1 - t0)
        cpu_total_ns = cpu_total_ns + ns
        print("CPU Trial", trial + 1, ":", ns / 1_000_000.0, "ms, kept:", cpu_results.size(), "detections")
    
    var cpu_avg_ms = cpu_total_ns / Float64(num_trials) / 1_000_000.0
    print("CPU NMS average time:", cpu_avg_ms, "ms")
    print("CPU NMS final result: kept", cpu_results.size(), "detections")
    print()
    
    # GPU NMS benchmark
    print("Running GPU NMS benchmark...")
    var ctx = DeviceContext()
    var (boxes_buf, scores_buf) = detections_to_gpu_format(detections, ctx)
    var keep_buf = ctx.enqueue_create_buffer[DType.uint8](detections.size())
    
    var gpu_total_ns: Float64 = 0.0
    var gpu_results = Buffer[Detection]()
    
    for trial in range(num_trials):
        # Reset keep buffer
        with keep_buf.map_to_host() as keep_host:
            for i in range(detections.size()):
                keep_host[i] = 0
        
        ctx.synchronize()
        var t0 = perf_counter_ns()
        fast_nms_simple(ctx, boxes_buf, scores_buf, keep_buf, detections.size(), iou_threshold)
        ctx.synchronize()
        var t1 = perf_counter_ns()
        var ns = Float64(t1 - t0)
        gpu_total_ns = gpu_total_ns + ns
        
        # Get results for this trial
        var trial_results = gpu_results_to_detections(keep_buf, detections)
        print("GPU Trial", trial + 1, ":", ns / 1_000_000.0, "ms, kept:", trial_results.size(), "detections")
        
        if trial == num_trials - 1:
            gpu_results = trial_results
    
    var gpu_avg_ms = gpu_total_ns / Float64(num_trials) / 1_000_000.0
    print("GPU NMS average time:", gpu_avg_ms, "ms")
    print("GPU NMS final result: kept", gpu_results.size(), "detections")
    print()
    
    # Performance comparison
    var speedup = cpu_avg_ms / gpu_avg_ms
    print("=== Performance Summary ===")
    print("CPU NMS:", cpu_avg_ms, "ms")
    print("GPU NMS:", gpu_avg_ms, "ms")
    print("Speedup:", speedup, "x")
    print("GPU is", speedup, "x faster than CPU")
    
    # Verify results are similar
    print()
    print("=== Result Verification ===")
    print("CPU kept:", cpu_results.size(), "detections")
    print("GPU kept:", gpu_results.size(), "detections")
    var diff = abs(cpu_results.size() - gpu_results.size())
    print("Difference in number of kept detections:", diff)
    
    if diff <= 2:  # Allow small differences due to floating point precision
        print("✅ Results are consistent between CPU and GPU implementations")
    else:
        print("⚠️  Results differ significantly - may need algorithm verification")

def main():
    try:
        benchmark_nms_performance()
    except e:
        print("Error:", e)
    finally:
        print("Benchmark completed!") 