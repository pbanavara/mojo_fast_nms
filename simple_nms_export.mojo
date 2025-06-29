from python import Python, PythonObject
from gpu.host import DeviceContext, DeviceBuffer
from warp_bitmask_nms_simple import fast_nms_simple

fn simple_nms_export(boxes: PythonObject, scores: PythonObject, iou_threshold: PythonObject) raises -> PythonObject:
    """Simple NMS export that takes Python lists and returns Python list of indices."""
    
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

# Test function
fn main() raises:
    print("Testing simple NMS export...")
    
    # Create test data
    var test_boxes = Python.list()
    var test_scores = Python.list()
    
    # Add some test boxes
    for i in range(10):
        test_boxes.append(Float32(i * 10))
        test_boxes.append(Float32(i * 10))
        test_boxes.append(Float32(i * 10 + 50))
        test_boxes.append(Float32(i * 10 + 50))
        test_scores.append(Float32(0.9 - i * 0.1))
    
    print("Created test data")
    
    # Run NMS
    var result = simple_nms_export(test_boxes, test_scores, 0.5)
    print("NMS result:", result)
    print("Test completed successfully!") 