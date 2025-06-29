from python import Python, PythonObject
from gpu.host import DeviceContext, DeviceBuffer
from warp_bitmask_nms_simple import fast_nms_simple

fn test_nms_with_python() raises:
    print("Testing NMS with Python interop...")
    
    # Create test data using Python
    var boxes = Python.list()
    var scores = Python.list()
    
    # Add 10 test boxes
    for i in range(10):
        # Each box has 4 coordinates: [y1, x1, y2, x2]
        boxes.append(Float32(i * 10))  # y1
        boxes.append(Float32(i * 10))  # x1
        boxes.append(Float32(i * 10 + 50))  # y2
        boxes.append(Float32(i * 10 + 50))  # x2
        scores.append(Float32(0.9 - i * 0.1))
    
    print("Created test data with", scores.len(), "boxes")
    
    # Convert to Mojo types
    var n = scores.len()
    var iou_threshold = Float32(0.5)
    
    # Create GPU context
    var ctx = DeviceContext()
    
    # Allocate GPU buffers
    var boxes_buf = ctx.enqueue_create_buffer[DType.float32](n * 4)
    var scores_buf = ctx.enqueue_create_buffer[DType.float32](n)
    var keep_buf = ctx.enqueue_create_buffer[DType.uint8](n)
    
    # Copy data to GPU
    with boxes_buf.map_to_host() as boxes_host:
        for i in range(n * 4):
            boxes_host[i] = Float32(boxes[i].to_float())
    
    with scores_buf.map_to_host() as scores_host:
        for i in range(n):
            scores_host[i] = Float32(scores[i].to_float())
    
    # Run NMS
    fast_nms_simple(ctx, boxes_buf, scores_buf, keep_buf, n, iou_threshold)
    
    # Get results back
    var kept_count = 0
    with keep_buf.map_to_host() as keep_host:
        for i in range(n):
            if keep_host[i] != 0:
                kept_count += 1
    
    print("NMS completed. Kept", kept_count, "out of", n, "boxes")
    
    # Convert results back to Python
    var result = Python.list()
    with keep_buf.map_to_host() as keep_host:
        for i in range(n):
            if keep_host[i] != 0:
                result.append(i)
    
    print("Result indices:", result)
    print("Test completed successfully!")

fn main() raises:
    test_nms_with_python() 