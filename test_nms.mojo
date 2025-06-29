from gpu.host import DeviceContext, DeviceBuffer
from warp_bitmask_nms import fast_nms, TILE

def main():
    try:
        print("Testing warp_bitmask_nms module...")
        print("Module imported successfully!")
        print("TILE size:", TILE)
        
        # Test with 4 boxes that have overlapping regions
        n = 4
        
        # Create device context
        var ctx = DeviceContext()
        
        # Create device buffers
        var boxes_buf = ctx.enqueue_create_buffer[DType.float32](n * 4)
        var scores_buf = ctx.enqueue_create_buffer[DType.float32](n)
        var keep_buf = ctx.enqueue_create_buffer[DType.uint8](n)
        
        # Initialize test data as flat arrays
        with boxes_buf.map_to_host() as boxes_host:
            # Box 0: [0.0, 0.0, 1.0, 1.0] - score 0.95
            boxes_host[0*4+0] = 0.0
            boxes_host[0*4+1] = 0.0
            boxes_host[0*4+2] = 1.0
            boxes_host[0*4+3] = 1.0
            # Box 1: [0.1, 0.1, 1.1, 1.1] - score 0.9 (overlaps with box 0)
            boxes_host[1*4+0] = 0.1
            boxes_host[1*4+1] = 0.1
            boxes_host[1*4+2] = 1.1
            boxes_host[1*4+3] = 1.1
            # Box 2: [2.0, 2.0, 3.0, 3.0] - score 0.8 (no overlap)
            boxes_host[2*4+0] = 2.0
            boxes_host[2*4+1] = 2.0
            boxes_host[2*4+2] = 3.0
            boxes_host[2*4+3] = 3.0
            # Box 3: [0.9, 0.9, 1.9, 1.9] - score 0.7 (overlaps with box 0)
            boxes_host[3*4+0] = 0.9
            boxes_host[3*4+1] = 0.9
            boxes_host[3*4+2] = 1.9
            boxes_host[3*4+3] = 1.9
        with scores_buf.map_to_host() as scores_host:
            scores_host[0] = 0.95
            scores_host[1] = 0.9
            scores_host[2] = 0.8
            scores_host[3] = 0.7
        
        print("Running NMS with IoU threshold: 0.5")
        print("Expected: Box 0 (score 0.95) and Box 2 (score 0.8) should be kept")
        
        # Run the NMS kernel
        fast_nms(ctx, boxes_buf, scores_buf, keep_buf, n, 0.5)
        
        # Read back results as flat array
        with keep_buf.map_to_host() as keep_host:
            keep0 = keep_host[0]
            keep1 = keep_host[1]
            keep2 = keep_host[2]
            keep3 = keep_host[3]
        
        print("NMS Results:")
        print("Box 0 (score 0.95) keep:", keep0)
        print("Box 1 (score 0.90) keep:", keep1)
        print("Box 2 (score 0.80) keep:", keep2)
        print("Box 3 (score 0.70) keep:", keep3)
        
        # Validate results
        if keep0 == 1 and keep1 == 0 and keep2 == 1 and keep3 == 1:
            print("✅ Test PASSED: NMS correctly kept highest scoring non-overlapping boxes")
        else:
            print("❌ Test FAILED: Unexpected NMS results")
            print("Expected: Box 0=1, Box 1=0, Box 2=1, Box 3=1")
            print("Got: Box 0=", keep0, ", Box 1=", keep1, ", Box 2=", keep2, ", Box 3=", keep3)
        
        print("Test completed!")
    except e:
        print("Error:", e)
    finally:
        print("Done")