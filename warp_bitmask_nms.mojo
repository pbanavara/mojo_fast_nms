from gpu        import thread_idx, block_idx, block_dim, barrier
from gpu.host   import DeviceContext, DeviceBuffer
from gpu.sync import syncwarp
from gpu.memory import AddressSpace
from memory import stack_allocation, UnsafePointer

alias TILE = 32

def tiles_for(n: Int) -> Int:
    return (n + TILE - 1) // TILE

# Kernel for building suppression in upper matrix - simplified to match CUDA
fn nms_bitmask_kernel(
    boxes: UnsafePointer[Float32],      # [N, 4] - boxes in format (y1, x1, y2, x2)
    scores: UnsafePointer[Float32],     # [N] - scores (already DESC sorted)
    mask: UnsafePointer[UInt32],        # [N, ceil(N/32)] - bitmask matrix
    n: Int,                             # number of boxes
    iou_th: Float32                     # IoU threshold
) raises:
    # Global coordinates of this thread in the comparison matrix
    var row = block_idx.y * TILE + thread_idx.y     # box i
    var col = block_idx.x * TILE + thread_idx.x     # box j

    if row >= n or col >= n or col >= row:          # keep upper‑tri only
        return

    # Stage 32 row‑boxes into shared memory (one per threadIdx.y)
    var sb = stack_allocation[
            TILE * 4,
            Float32,
            address_space = AddressSpace.SHARED,
    ]()

    if thread_idx.x == 0:
        for i in range(0, 4):
            sb[thread_idx.y * 4 + i] = boxes[row * 4 + i]

    barrier()

    # Get peer box coordinates
    var y1a = sb[thread_idx.y * 4 + 0]
    var x1a = sb[thread_idx.y * 4 + 1]
    var y2a = sb[thread_idx.y * 4 + 2]
    var x2a = sb[thread_idx.y * 4 + 3]
    
    var y1b = boxes[col * 4 + 0]
    var x1b = boxes[col * 4 + 1]
    var y2b = boxes[col * 4 + 2]
    var x2b = boxes[col * 4 + 3]
    
    # Calculate intersection
    var inter_y1 = y1a if y1a > y1b else y1b
    var inter_x1 = x1a if x1a > x1b else x1b
    var inter_y2 = y2a if y2a < y2b else y2b
    var inter_x2 = x2a if x2a < x2b else x2b
    
    var inter_h = inter_y2 - inter_y1
    var inter_w = inter_x2 - inter_x1
    var inter = Float32(0)
    if inter_h > 0 and inter_w > 0:
        inter = inter_h * inter_w
    
    # Calculate areas and IoU
    var area_a = (y2a - y1a) * (x2a - x1a)
    var area_b = (y2b - y1b) * (x2b - x1b)
    var uni = area_a + area_b - inter
    var iou_val = inter / uni
    
    # Determine if boxes overlap
    var overlaps: UInt32 = 0
    if iou_val >= iou_th:
        overlaps = 1
    
    syncwarp()
    # Use ballot operation to collect overlap information from all threads in warp
    var ballot: UInt32 = 0
    # In a real GPU implementation, this would use __ballot_sync or similar
    # For now, we'll use a simple approach where each thread sets its bit
    if overlaps == 1:
        ballot = 1 << thread_idx.x
    
    var dst = row * tiles_for(n) + col // TILE
    mask[dst] = ballot << (col & 31)

# Kernel for sweeping the mask - simplified to match CUDA
fn sweep_mask_kernel(
    n: Int,                              # number of boxes
    mask: UnsafePointer[UInt32],         # [N, ceil(N/32)] - bitmask matrix
    keep: UnsafePointer[UInt8]           # [N] - output: 1 if box should be kept, 0 otherwise
) raises:
    var gid = block_idx.x * block_dim.x + thread_idx.x
    if gid >= n: return

    # Walk previous kept boxes in 32‑bit chunks
    var alive: UInt8 = 1
    for c in range(0, tiles_for(n)):
        if mask[gid * tiles_for(n) + c] != 0:
            alive = 0
            break
    keep[gid] = alive

# Host convenience wrapper (enqueue both kernels) - simplified
fn fast_nms(ctx: DeviceContext,
    boxes_buf: DeviceBuffer[DType.float32], 
    scores_buf: DeviceBuffer[DType.float32], 
    keep_buf: DeviceBuffer[DType.uint8], 
    n: Int,
    iou_th: Float32) raises:

    var tiles = tiles_for(n)
    var grid  = (tiles, tiles)
    var block = (TILE, TILE)

    var mask_buf = ctx.enqueue_create_buffer[DType.uint32](n * tiles)
    _ = mask_buf.enqueue_fill(0)

    ctx.enqueue_function[nms_bitmask_kernel](
        boxes_buf.unsafe_ptr(), scores_buf.unsafe_ptr(), mask_buf.unsafe_ptr(), n, iou_th,
        grid_dim  = grid,
        block_dim = block
    )

    # Sweep (1‑D launch)
    var sweep_grid  = ((n + TILE - 1) // TILE,)
    var sweep_block = (TILE,)

    ctx.enqueue_function[sweep_mask_kernel](
        n, mask_buf.unsafe_ptr(), keep_buf.unsafe_ptr(),
        grid_dim  = sweep_grid,
        block_dim = sweep_block
    )