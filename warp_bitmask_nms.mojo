from gpu        import thread_idx, block_idx, block_dim, barrier
from gpu.host   import DeviceContext, DeviceBuffer
from layout     import Layout, LayoutTensor
from math import ceil 
from gpu.sync import syncwarp
from gpu.memory import AddressSpace
from memory import stack_allocation

alias TILE=32

fn tiles_for(n: Int) -> Int:
    return (n + TILE - 1)

alias BoxLayout   = Layout.row_major(4)
alias ScoreLayout = Layout.row_major(1)

fn MaskLayout_for(n: Int) -> Layout:
    var chunks: Int = (n + TILE - 1) // TILE
    return Layout.row_major(chunks)

@value
struct BoundingBox[T: DType]:
    var nw: SIMD[T, 2]
    var se: SIMD[T, 2]

    fn __init__(out self, 
        y1: Scalar[T], 
        x1: Scalar[T], 
        y2: Scalar[T], 
        x2: Scalar[T]):
        self.nw = SIMD[T, 2](max(y1, y2), max(x1, x2))
        self.se = SIMD[T, 2](min(y1, y2), min(x1, x2))

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
    scores: LayoutTensor[T,        ScoreLayout],    # [N,1] (already DESC sorted)
    mask:   LayoutTensor[mut=True, DType.uint32 ],# [N, ceil(N/32)]
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
            TILE,                # number of BoundingBox elements
            BoundingBox[T],      # element type (generic over T)
            address_space = AddressSpace.SHARED,
    ]()

    if thread_idx.x == 0:   # one thread per row loads
        var bbox: BoundingBox[T] = BoundingBox(
            boxes[row, 0], boxes[row, 1], boxes[row, 2], boxes[row, 3]
        )
        sb[thread_idx.y] = bbox

    barrier()          # wait until shared is populated

    # Load peer box directly from global (col)
    var peer_bb = BoundingBox(
        boxes[col, 0], boxes[col, 1], boxes[col, 2], boxes[col, 3]
    )

    # ------------------------------------------------------------------
    # Compute IoU   (row = sb[threadIdx.y], col = peer_bb)
    # ------------------------------------------------------------------
    var overlaps = sb[thread_idx.y].iou(peer_bb) >= iou_th.cast[T]() ? 1 : 0

    # ------------------------------------------------------------------
    # Warp‑wide ballot: pack 32 overlaps into one 32‑bit integer
    # ------------------------------------------------------------------
    #var ballot: UInt32 = ballot_sync(0xffffffff, overlaps)    
    syncwarp()

    # TODO: adjust intrinsic if different


    # Store mask word for (row, colChunk)
    var chunk = col / TILE                               # which 32‑column chunk this col belongs to
    var dst   = row * ((n + TILE - 1) // TILE) + chunk
    simt.atomic_or(mask.raw_ptr()[dst], ballot << (col & 31))

fn sweep_mask_kernel(
    n:         Int,
    mask:      LayoutTensor[DType.uint32, MaskLayout_for(n)],
    keep:      LayoutTensor[mut=True, DType.uint8, Layout.row_major(1)], 
):
    var gid = block_idx.x * block_dim.x + thread_idx.x
    if gid >= n: return

    # Walk previous kept boxes in 32‑bit chunks
    var alive: UInt8 = 1
    for var c in 0 .. (n + TILE - 1) // TILE:
        if mask[gid, c] != 0:   # any bit -> suppressed
            alive = 0
            break
    keep[gid, 0] = alive

# -----------------------------------------------------------------------------
#  Host convenience wrapper (enqueue both kernels)
# -----------------------------------------------------------------------------
fn fast_nms[T: DType](ctx: DeviceContext,
    boxes_buf: DeviceBuffer[T], scores_buf: DeviceBuffer[T], 
    keep_buf: DeviceBuffer[DType.uint8], n: Int,
    iou_th: Float32):

    var tiles = tiles_for(n)
    var grid  = (tiles, tiles)
    var block = (TILE, TILE)

    var mask_buf = ctx.enqueue_create_buffer[DType.uint32](n * tiles).enqueue_fill(0)
    var mask_t   = LayoutTensor[mut=True, DType.uint32, Layout.row_major(tiles)](mask_buf.unsafe_ptr())

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
    var keep_t = LayoutTensor[mut=True, UInt8, Layout.row_major(1)](keep_buf.unsafe_ptr())

    ctx.enqueue_function[sweep_mask_kernel](
        mask_t, keep_t, n,
        grid_dim  = sweep_grid,
        block_dim = sweep_block
    )