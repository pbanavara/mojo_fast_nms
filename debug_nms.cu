#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TILE 32
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Calculate IoU between two boxes
__device__ float calculate_iou(float y1a, float x1a, float y2a, float x2a,
                              float y1b, float x1b, float y2b, float x2b) {
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
    
    float area_a = (y2a - y1a) * (x2a - x1a);
    float area_b = (y2b - y1b) * (x2b - x1b);
    float uni = area_a + area_b - inter;
    
    return inter / uni;
}

// Debug kernel to print IoU values
__global__ void debug_iou_kernel(
    const float* boxes,      // [N, 4] - boxes in format (y1, x1, y2, x2)
    int n,                   // number of boxes
    float iou_th             // IoU threshold
) {
    int row = blockIdx.y * TILE + threadIdx.y;  // box i
    int col = blockIdx.x * TILE + threadIdx.x;  // box j
    
    if (row >= n || col >= n || col >= row) {  // keep upper-tri only
        return;
    }
    
    // Get box coordinates
    float y1a = boxes[row * 4 + 0];
    float x1a = boxes[row * 4 + 1];
    float y2a = boxes[row * 4 + 2];
    float x2a = boxes[row * 4 + 3];
    
    float y1b = boxes[col * 4 + 0];
    float x1b = boxes[col * 4 + 1];
    float y2b = boxes[col * 4 + 2];
    float x2b = boxes[col * 4 + 3];
    
    float iou = calculate_iou(y1a, x1a, y2a, x2a, y1b, x1b, y2b, x2b);
    
    if (iou > 0.0f) {
        printf("Box %d vs Box %d: IoU = %.4f, threshold = %.4f, suppress = %s\n", 
               row, col, iou, iou_th, (iou >= iou_th) ? "YES" : "NO");
    }
}

// Simple NMS kernel for debugging
__global__ void simple_nms_kernel(
    const float* boxes,      // [N, 4] - boxes in format (y1, x1, y2, x2)
    const float* scores,     // [N] - scores
    uint8_t* keep,           // [N] - output
    int n,                   // number of boxes
    float iou_th             // IoU threshold
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    // Each thread processes one box
    keep[i] = 1;  // Assume we keep this box initially
    
    // Check against all higher-scoring boxes
    for (int j = 0; j < i; j++) {
        if (keep[j] == 0) continue;  // Skip already suppressed boxes
        
        // Get box coordinates
        float y1a = boxes[i * 4 + 0];
        float x1a = boxes[i * 4 + 1];
        float y2a = boxes[i * 4 + 2];
        float x2a = boxes[i * 4 + 3];
        
        float y1b = boxes[j * 4 + 0];
        float x1b = boxes[j * 4 + 1];
        float y2b = boxes[j * 4 + 2];
        float x2b = boxes[j * 4 + 3];
        
        float iou = calculate_iou(y1a, x1a, y2a, x2a, y1b, x1b, y2b, x2b);
        
        if (iou >= iou_th) {
            keep[i] = 0;  // Suppress this box
            break;
        }
    }
}

int main() {
    int n = 4;
    
    // Create host buffers
    float* h_boxes = (float*)malloc(n * 4 * sizeof(float));
    float* h_scores = (float*)malloc(n * sizeof(float));
    uint8_t* h_keep = (uint8_t*)malloc(n * sizeof(uint8_t));
    
    // Initialize test data (same as before)
    h_boxes[0*4 + 0] = 0.0f;  // Box 0: y1
    h_boxes[0*4 + 1] = 0.0f;  // Box 0: x1
    h_boxes[0*4 + 2] = 1.0f;  // Box 0: y2
    h_boxes[0*4 + 3] = 1.0f;  // Box 0: x2
    
    h_boxes[1*4 + 0] = 0.1f;  // Box 1: y1
    h_boxes[1*4 + 1] = 0.1f;  // Box 1: x1
    h_boxes[1*4 + 2] = 1.1f;  // Box 1: y2
    h_boxes[1*4 + 3] = 1.1f;  // Box 1: x2
    
    h_boxes[2*4 + 0] = 2.0f;  // Box 2: y1
    h_boxes[2*4 + 1] = 2.0f;  // Box 2: x1
    h_boxes[2*4 + 2] = 3.0f;  // Box 2: y2
    h_boxes[2*4 + 3] = 3.0f;  // Box 2: x2
    
    h_boxes[3*4 + 0] = 0.9f;  // Box 3: y1
    h_boxes[3*4 + 1] = 0.9f;  // Box 3: x1
    h_boxes[3*4 + 2] = 1.9f;  // Box 3: y2
    h_boxes[3*4 + 3] = 1.9f;  // Box 3: x2
    
    h_scores[0] = 0.95f;
    h_scores[1] = 0.9f;
    h_scores[2] = 0.8f;
    h_scores[3] = 0.7f;
    
    printf("Box coordinates:\n");
    for (int i = 0; i < n; i++) {
        printf("Box %d: [%.1f, %.1f, %.1f, %.1f] score=%.2f\n", 
               i, h_boxes[i*4+0], h_boxes[i*4+1], h_boxes[i*4+2], h_boxes[i*4+3], h_scores[i]);
    }
    
    printf("\nRunning IoU debug kernel...\n");
    
    // Allocate device buffers
    float* d_boxes;
    uint8_t* d_keep;
    
    CHECK_CUDA(cudaMalloc(&d_boxes, n * 4 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_keep, n * sizeof(uint8_t)));
    
    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_boxes, h_boxes, n * 4 * sizeof(float), cudaMemcpyHostToDevice));
    
    // Run debug kernel
    dim3 grid(1, 1);
    dim3 block(TILE, TILE);
    debug_iou_kernel<<<grid, block>>>(d_boxes, n, 0.5f);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    
    printf("\nRunning simple NMS kernel...\n");
    
    // Run simple NMS kernel
    dim3 simple_grid((n + 255) / 256);
    dim3 simple_block(256);
    simple_nms_kernel<<<simple_grid, simple_block>>>(d_boxes, d_boxes, d_keep, n, 0.5f);
    CHECK_CUDA(cudaGetLastError());
    
    // Copy results back
    CHECK_CUDA(cudaMemcpy(h_keep, d_keep, n * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    
    // Print results
    printf("\nNMS Results:\n");
    for (int i = 0; i < n; i++) {
        printf("Box %d (score %.2f) keep: %d\n", i, h_scores[i], h_keep[i]);
    }
    
    // Validate results
    if (h_keep[0] == 1 && h_keep[1] == 0 && h_keep[2] == 1 && h_keep[3] == 0) {
        printf("✅ Test PASSED: NMS correctly kept highest scoring non-overlapping boxes\n");
    } else {
        printf("❌ Test FAILED: Unexpected NMS results\n");
    }
    
    // Clean up
    CHECK_CUDA(cudaFree(d_boxes));
    CHECK_CUDA(cudaFree(d_keep));
    free(h_boxes);
    free(h_scores);
    free(h_keep);
    
    printf("Test completed!\n");
    return 0;
} 