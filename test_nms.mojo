from gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor
from warp_bitmask_nms import fast_nms, TILE

alias Float32 = DType.float32
alias UInt8   = DType.uint8
alias BoxLayout = Layout.row_major(4)
alias ScoreLayout = Layout.row_major(1)

def main():
    try:
        print("Testing warp_bitmask_nms module...")
        print("Module imported successfully!")
        print("TILE size:", TILE)
        
        # Test with 4 boxes that have overlapping regions
        var n = 4
        
        # Create device context
        var ctx = DeviceContext()
        
        # Create device buffers
        var boxes_buf = ctx.enqueue_create_buffer[Float32](n * 4)
        var scores_buf = ctx.enqueue_create_buffer[Float32](n)
        var keep_buf = ctx.enqueue_create_buffer[UInt8](n)
        
        # Create LayoutTensors to access the data
        var boxes_t = LayoutTensor[Float32, BoxLayout](boxes_buf.unsafe_ptr())
        var scores_t = LayoutTensor[Float32, ScoreLayout](scores_buf.unsafe_ptr())
        var keep_t = LayoutTensor[mut=True, UInt8, Layout.row_major(1)](keep_buf.unsafe_ptr())
        
        # Initialize test data using LayoutTensor
        # Box 0: [0.0, 0.0, 1.0, 1.0] - score 0.95
        # Box 1: [0.1, 0.1, 1.1, 1.1] - score 0.9 (overlaps with box 0)
        # Box 2: [2.0, 2.0, 3.0, 3.0] - score 0.8 (no overlap)
        # Box 3: [0.9, 0.9, 1.9, 1.9] - score 0.7 (overlaps with box 0)
        
        boxes_t[0, 0] = 0.0  # Box 0: y1
        boxes_t[0, 1] = 0.0  # Box 0: x1  
        boxes_t[0, 2] = 1.0  # Box 0: y2
        boxes_t[0, 3] = 1.0  # Box 0: x2
        
        boxes_t[1, 0] = 0.1  # Box 1: y1
        boxes_t[1, 1] = 0.1  # Box 1: x1
        boxes_t[1, 2] = 1.1  # Box 1: y2
        boxes_t[1, 3] = 1.1  # Box 1: x2
        
        boxes_t[2, 0] = 2.0  # Box 2: y1
        boxes_t[2, 1] = 2.0  # Box 2: x1
        boxes_t[2, 2] = 3.0  # Box 2: y2
        boxes_t[2, 3] = 3.0  # Box 2: x2
        
        boxes_t[3, 0] = 0.9  # Box 3: y1
        boxes_t[3, 1] = 0.9  # Box 3: x1
        boxes_t[3, 2] = 1.9  # Box 3: y2
        boxes_t[3, 3] = 1.9  # Box 3: x2
        
        # Fill scores
        scores_t[0, 0] = 0.95
        scores_t[1, 0] = 0.9
        scores_t[2, 0] = 0.8
        scores_t[3, 0] = 0.7
        
        print("Running NMS with IoU threshold: 0.5")
        print("Expected: Box 0 (score 0.95) and Box 2 (score 0.8) should be kept")
        
        # Run the NMS kernel
        fast_nms[Float32](ctx, boxes_buf, scores_buf, keep_buf, n, 0.5)
        
        # Read back results using LayoutTensor
        var keep0 = keep_t[0, 0]
        var keep1 = keep_t[1, 0]
        var keep2 = keep_t[2, 0]
        var keep3 = keep_t[3, 0]
        
        print("NMS Results:")
        print("Box 0 (score 0.95) keep:", keep0)
        print("Box 1 (score 0.90) keep:", keep1)
        print("Box 2 (score 0.80) keep:", keep2)
        print("Box 3 (score 0.70) keep:", keep3)
        
        # Validate results
        if keep0 == 1 and keep1 == 0 and keep2 == 1 and keep3 == 0:
            print("✅ Test PASSED: NMS correctly kept highest scoring non-overlapping boxes")
        else:
            print("❌ Test FAILED: Unexpected NMS results")
        
        print("Test completed!")
    except e:
        print("Error:", e)
    finally:
        print("Done")