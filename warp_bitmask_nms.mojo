from gpu        import thread_idx, block_idx, block_dim, barrier
from gpu.host   import DeviceContext, DeviceBuffer
from layout     import Layout, LayoutTensor
from math import ceil 
from gpu.sync import syncwarp
from gpu.memory import AddressSpace
from memory import stack_allocation
# from gpu.atomic import atomic_or  # Not available in this Mojo version

alias TILE=32

fn tiles_for(n: Int) -> Int:
    return (n + TILE - 1) // TILE

alias BoxLayout   = Layout.row_major(4)
alias ScoreLayout = Layout.row_major(1)
alias MaskLayout = Layout.row_major(1)

@value
struct BoundingBox[T: DType]:
    var nw: SIMD[T, 2]
    var se: SIMD[T, 2]

    fn area(self) -> Scalar[T]:
        return (self.se[0] - self.nw[0]) * (self.se[1] - self.nw[1])

    fn intersection(self, other: Self) -> Scalar[T]:
        var nw = min(self.nw, other.nw)
        var se = max(self.se, other.se)
        if nw[1] < se[1] or nw[0] < se[0]:
            return 0
        return Self(nw, se).area()

    fn iou(self, other: Self) -> Scalar[T]:
        var inter = self.intersection(other)
        var uni   = self.area() + other.area() - inter
        return abs(inter) / abs(uni)

#Kernel for building suppression in upper matrix
fn nms_bitmask_kernel[T: DType](
    boxes:  LayoutTensor[T,        BoxLayout],      # [N,4]
    scores: LayoutTensor[T,        ScoreLayout],    # [N,4] (already DESC sorted)
    mask:   LayoutTensor[mut=True, DType.uint32, MaskLayout],# [N, ceil(N/32)]
    n:      Int,
    iou_th: Float32
):
    # Global coordinates of this thread in the comparison matrix
    var row = block_idx.y * TILE + thread_idx.y     # box i
    var col = block_idx.x * TILE + thread_idx.x     # box j

    if row >= n or col >= n or col >= row:          # keep upper‑tri only
        return

    # ------------------------------------------------------------------
    # Stage 32 row‑boxes into shared memory (one per threadIdx.y)
    # ------------------------------------------------------------------
    var sb = stack_allocation[
            TILE * 4,
            T,
            address_space = AddressSpace.SHARED,
    ]()

    if thread_idx.x == 0:
        for i in range(0, 4):
            sb[thread_idx.y * 4 + i] = boxes[row, i][0]

    barrier()

    var peer_box = (boxes[col, 0][0], boxes[col, 1][0], boxes[col, 2][0], boxes[col, 3][0])

    var row_box = (sb[Int(thread_idx.y) * 4 + 0], sb[Int(thread_idx.y) * 4 + 1], sb[Int(thread_idx.y) * 4 + 2], sb[Int(thread_idx.y) * 4 + 3])
    var y1a = row_box[0]
    var x1a = row_box[1]
    var y2a = row_box[2]
    var x2a = row_box[3]
    var y1b = peer_box[0]
    var x1b = peer_box[1]
    var y2b = peer_box[2]
    var x2b = peer_box[3]
    var inter_y1 = y1a if y1a > y1b else y1b
    var inter_x1 = x1a if x1a > x1b else x1b
    var inter_y2 = y2a if y2a < y2b else y2b
    var inter_x2 = x2a if x2a < x2b else x2b
    var inter_h = inter_y2 - inter_y1
    var inter_w = inter_x2 - inter_x1
    var inter = SIMD[T, 1](0)
    if inter_h > 0 and inter_w > 0:
        inter = inter_h * inter_w
    var area_a = (y2a - y1a) * (x2a - x1a)
    var area_b = (y2b - y1b) * (x2b - x1b)
    var uni = area_a + area_b - inter
    var iou_val = abs(inter) / abs(uni)
    var overlaps: UInt32
    if iou_val >= iou_th.cast[T]():
        overlaps = 1
    else:
        overlaps = 0

    syncwarp()
    # Use ballot operation to collect overlap information from all threads in warp
    var ballot: UInt32 = 0
    # In a real GPU implementation, this would use __ballot_sync or similar
    # For now, we'll use a simple approach where each thread sets its bit
    if overlaps == 1:
        ballot = 1 << thread_idx.x
    
    var dst = Int(row) * ((n + TILE - 1) // TILE) + Int(col / TILE)
    mask[dst] = ballot << (col & 31)

fn sweep_mask_kernel(
    n:         Int,
    mask:      LayoutTensor[DType.uint32, Layout.row_major(1)],
    keep:      LayoutTensor[mut=True, DType.uint8, Layout.row_major(1)]
):
    var gid = block_idx.x * block_dim.x + thread_idx.x
    if gid >= n: return

    # Walk previous kept boxes in 32‑bit chunks
    var alive: UInt8 = 1
    for c in range(0, (n + TILE - 1) // TILE):
        if mask[gid * ((n + TILE - 1) // TILE) + c] != 0:
            alive = 0
            break
    keep[gid] = alive

# -----------------------------------------------------------------------------
#  Host convenience wrapper (enqueue both kernels)
# -----------------------------------------------------------------------------
fn fast_nms[T: DType](ctx: DeviceContext,
    boxes_buf: DeviceBuffer[T], scores_buf: DeviceBuffer[T], 
    keep_buf: DeviceBuffer[DType.uint8], n: Int,
    iou_th: Float32) raises:

    var tiles = tiles_for(n)
    var grid  = (tiles, tiles)
    var block = (TILE, TILE)

    var mask_buf = ctx.enqueue_create_buffer[DType.uint32](n * tiles)
    _ = mask_buf.enqueue_fill(0)
    var mask_t   = LayoutTensor[mut=True, DType.uint32, Layout.row_major(1)](mask_buf.unsafe_ptr())

    var boxes_t  = LayoutTensor[T, BoxLayout](boxes_buf.unsafe_ptr())
    var scores_t = LayoutTensor[T, ScoreLayout](scores_buf.unsafe_ptr())

    ctx.enqueue_function[nms_bitmask_kernel](
        boxes_t, scores_t, mask_t, n, iou_th,
        grid_dim  = grid,
        block_dim = block
    )

    # Sweep   (1‑D launch)
    var sweep_grid  = ((n + TILE - 1) // TILE,)
    var sweep_block = (TILE,)
    var keep_t = LayoutTensor[mut=True, DType.uint8, Layout.row_major(1)](keep_buf.unsafe_ptr())

    ctx.enqueue_function[sweep_mask_kernel](
        n, mask_t, keep_t,
        grid_dim  = sweep_grid,
        block_dim = sweep_block
    )