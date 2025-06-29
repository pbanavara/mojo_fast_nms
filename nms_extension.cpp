#include <torch/extension.h>
#include <vector>

// Forward declaration of the CUDA kernel
extern "C" void fast_nms(
    const float* boxes_buf,      // [N, 4] - boxes in device memory
    const float* scores_buf,     // [N] - scores in device memory
    uint8_t* keep_buf,           // [N] - output in device memory
    int n,                       // number of boxes
    float iou_th                 // IoU threshold
);

// PyTorch binding function
torch::Tensor fast_nms_torch(torch::Tensor boxes, torch::Tensor scores, float iou_th) {
    // Check input tensors
    TORCH_CHECK(boxes.is_cuda(), "boxes must be a CUDA tensor");
    TORCH_CHECK(scores.is_cuda(), "scores must be a CUDA tensor");
    TORCH_CHECK(boxes.dim() == 2, "boxes must be 2D tensor [N, 4]");
    TORCH_CHECK(boxes.size(1) == 4, "boxes must have 4 columns [x1, y1, x2, y2]");
    TORCH_CHECK(scores.dim() == 1, "scores must be 1D tensor [N]");
    TORCH_CHECK(boxes.size(0) == scores.size(0), "boxes and scores must have same number of rows");
    
    int n = boxes.size(0);
    
    // Create output tensor for keep mask
    auto keep = torch::zeros({n}, torch::dtype(torch::kUInt8).device(boxes.device()));
    
    // Call the CUDA kernel
    fast_nms(
        boxes.data_ptr<float>(),
        scores.data_ptr<float>(),
        keep.data_ptr<uint8_t>(),
        n,
        iou_th
    );
    
    return keep;
}

// Module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fast_nms", &fast_nms_torch, "Fast NMS (CUDA)",
          py::arg("boxes"), py::arg("scores"), py::arg("iou_threshold"));
} 