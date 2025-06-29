#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define TILE 32
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Calculate number of tiles needed
__host__ __device__ int tiles_for(int n) {
    return (n + TILE - 1) / TILE;
}

// Bounding box structure
struct BoundingBox {
    float nw[2];  // northwest corner (x, y)
    float se[2];  // southeast corner (x, y)
    
    __device__ float area() const {
        return (se[0] - nw[0]) * (se[1] - nw[1]);
    }
    
    __device__ float intersection(const BoundingBox& other) const {
        float nw_x = max(nw[0], other.nw[0]);
        float nw_y = max(nw[1], other.nw[1]);
        float se_x = min(se[0], other.se[0]);
        float se_y = min(se[1], other.se[1]);
        
        if (nw_x >= se_x || nw_y >= se_y) {
            return 0.0f;
        }
        return (se_x - nw_x) * (se_y - nw_y);
    }
    
    __device__ float iou(const BoundingBox& other) const {
        float inter = intersection(other);
        float uni = area() + other.area() - inter;
        return inter / uni;
    }
};

// Kernel for building suppression in upper matrix
__global__ void nms_bitmask_kernel(
    const float* boxes,      // [N, 4] - boxes in format (y1, x1, y2, x2)
    const float* scores,     // [N] - scores (already DESC sorted)
    uint32_t* mask,          // [N, ceil(N/32)] - bitmask matrix
    int n,                   // number of boxes
    float iou_th             // IoU threshold
) {
    // Global coordinates of this thread in the comparison matrix
    int row = blockIdx.y * TILE + threadIdx.y;  // box i
    int col = blockIdx.x * TILE + threadIdx.x;  // box j
    
    if (row >= n || col >= n || col >= row) {  // keep upper-tri only
        return;
    }
    
    // Stage 32 row-boxes into shared memory (one per threadIdx.y)
    __shared__ float sb[TILE * 4];
    
    if (threadIdx.x == 0) {
        for (int i = 0; i < 4; i++) {
            sb[threadIdx.y * 4 + i] = boxes[row * 4 + i];
        }
    }
    __syncthreads();
    
    // Get peer box coordinates
    float y1a = sb[threadIdx.y * 4 + 0];
    float x1a = sb[threadIdx.y * 4 + 1];
    float y2a = sb[threadIdx.y * 4 + 2];
    float x2a = sb[threadIdx.y * 4 + 3];
    
    float y1b = boxes[col * 4 + 0];
    float x1b = boxes[col * 4 + 1];
    float y2b = boxes[col * 4 + 2];
    float x2b = boxes[col * 4 + 3];
    
    // Calculate intersection
    float inter_y1 = max(y1a, y1b);
    float inter_x1 = max(x1a, x1b);
    float inter_y2 = min(y2a, y2b);
    float inter_x2 = min(x2a, x2b);
    
    float inter_h = inter_y2 - inter_y1;
    float inter_w = inter_x2 - inter_x1;
    float inter = 0.0f;
    if (inter_h > 0 && inter_w > 0) {
        inter = inter_h * inter_w;
    }
    
    // Calculate areas and IoU
    float area_a = (y2a - y1a) * (x2a - x1a);
    float area_b = (y2b - y1b) * (x2b - x1b);
    float uni = area_a + area_b - inter;
    float iou_val = inter / uni;
    
    // Determine if boxes overlap
    uint32_t overlaps = (iou_val >= iou_th) ? 1 : 0;
    
    // Use ballot operation to collect overlap information from all threads in warp
    uint32_t ballot = __ballot_sync(0xFFFFFFFF, overlaps);
    
    // Write to mask matrix
    int dst = row * tiles_for(n) + col / TILE;
    mask[dst] = ballot << (col & 31);
}

// Kernel for sweeping the mask to determine which boxes to keep
__global__ void sweep_mask_kernel(
    int n,                   // number of boxes
    const uint32_t* mask,    // [N, ceil(N/32)] - bitmask matrix
    uint8_t* keep            // [N] - output: 1 if box should be kept, 0 otherwise
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    
    // Walk previous kept boxes in 32-bit chunks
    uint8_t alive = 1;
    for (int c = 0; c < tiles_for(n); c++) {
        if (mask[gid * tiles_for(n) + c] != 0) {
            alive = 0;
            break;
        }
    }
    keep[gid] = alive;
}

// Host convenience wrapper (enqueue both kernels)
extern "C"
void fast_nms(
    const float* boxes_buf,      // [N, 4] - boxes in device memory
    const float* scores_buf,     // [N] - scores in device memory
    uint8_t* keep_buf,           // [N] - output in device memory
    int n,                       // number of boxes
    float iou_th                 // IoU threshold
) {
    int tiles = tiles_for(n);
    dim3 grid(tiles, tiles);
    dim3 block(TILE, TILE);
    
    // Allocate mask buffer
    uint32_t* mask_buf;
    CHECK_CUDA(cudaMalloc(&mask_buf, n * tiles * sizeof(uint32_t)));
    CHECK_CUDA(cudaMemset(mask_buf, 0, n * tiles * sizeof(uint32_t)));
    
    // Launch first kernel
    nms_bitmask_kernel<<<grid, block>>>(
        boxes_buf, scores_buf, mask_buf, n, iou_th
    );
    CHECK_CUDA(cudaGetLastError());
    
    // Launch second kernel (1D)
    dim3 sweep_grid((n + TILE - 1) / TILE);
    dim3 sweep_block(TILE);
    
    sweep_mask_kernel<<<sweep_grid, sweep_block>>>(
        n, mask_buf, keep_buf
    );
    CHECK_CUDA(cudaGetLastError());
    
    // Clean up
    CHECK_CUDA(cudaFree(mask_buf));
}

// Test function
int main() {
    int n = 4096;
    int num_trials = 10;
    
    // Create host buffers
    float* h_boxes = (float*)malloc(n * 4 * sizeof(float));
    float* h_scores = (float*)malloc(n * sizeof(float));
    uint8_t* h_keep = (uint8_t*)malloc(n * sizeof(uint8_t));
    
    // Initialize random test data
    srand(42);
    for (int i = 0; i < n; i++) {
        float y1 = (float)(rand() % 1000) / 1000.0f;
        float x1 = (float)(rand() % 1000) / 1000.0f;
        float h = (float)(rand() % 100) / 1000.0f + 0.05f;
        float w = (float)(rand() % 100) / 1000.0f + 0.05f;
        float y2 = y1 + h;
        float x2 = x1 + w;
        h_boxes[i*4+0] = y1;
        h_boxes[i*4+1] = x1;
        h_boxes[i*4+2] = y2;
        h_boxes[i*4+3] = x2;
        h_scores[i] = (float)(rand() % 1000) / 1000.0f;
    }
    
    // Allocate device buffers
    float* d_boxes;
    float* d_scores;
    uint8_t* d_keep;
    CHECK_CUDA(cudaMalloc(&d_boxes, n * 4 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_scores, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_keep, n * sizeof(uint8_t)));
    
    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_boxes, h_boxes, n * 4 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_scores, h_scores, n * sizeof(float), cudaMemcpyHostToDevice));
    
    // Benchmark
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    float total_ms = 0.0f;
    for (int trial = 0; trial < num_trials; ++trial) {
        CHECK_CUDA(cudaMemset(d_keep, 0, n * sizeof(uint8_t)));
        CHECK_CUDA(cudaEventRecord(start));
        fast_nms(d_boxes, d_scores, d_keep, n, 0.5f);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        total_ms += ms;
    }
    printf("\nCUDA NMS (n=%d, trials=%d): Avg time = %.3f ms\n", n, num_trials, total_ms / num_trials);
    
    // Clean up
    CHECK_CUDA(cudaFree(d_boxes));
    CHECK_CUDA(cudaFree(d_scores));
    CHECK_CUDA(cudaFree(d_keep));
    free(h_boxes);
    free(h_scores);
    free(h_keep);
    return 0;
} 