#!/usr/bin/env python3
"""
Test script for Mojo NMS integration with PyTorch
"""

import numpy as np
import torch
import time

def test_mojo_nms_integration():
    """Test Mojo NMS integration with PyTorch tensors."""
    
    print("Testing Mojo NMS integration...")
    
    # Create test data using PyTorch
    num_boxes = 1000
    boxes = torch.rand(num_boxes, 4)  # [x1, y1, x2, y2] format
    scores = torch.rand(num_boxes)
    
    # Sort by scores (descending)
    sorted_indices = torch.argsort(scores, descending=True)
    boxes = boxes[sorted_indices]
    scores = scores[sorted_indices]
    
    print(f"Created {num_boxes} test boxes with scores")
    print(f"Boxes shape: {boxes.shape}")
    print(f"Scores shape: {scores.shape}")
    
    # Convert to lists for Mojo (if needed)
    boxes_list = boxes.cpu().numpy().flatten().tolist()
    scores_list = scores.cpu().numpy().tolist()
    
    print("Data converted to Python lists")
    
    # Here we would call the Mojo NMS kernel
    # For now, let's simulate what the interface would look like
    print("\nMojo NMS interface would be:")
    print("mojo_nms_kernel(boxes_list, scores_list, iou_threshold=0.5)")
    
    # Simulate timing
    start_time = time.time()
    # result = mojo_nms_kernel(boxes_list, scores_list, 0.5)
    time.sleep(0.001)  # Simulate Mojo NMS execution
    end_time = time.time()
    
    print(f"Simulated Mojo NMS took {(end_time - start_time) * 1000:.3f} ms")
    
    # Compare with PyTorch CPU NMS (if available)
    try:
        from torchvision.ops import nms
        start_time = time.time()
        torch_result = nms(boxes, scores, 0.5)
        end_time = time.time()
        print(f"PyTorch CPU NMS took {(end_time - start_time) * 1000:.3f} ms")
        print(f"PyTorch kept {len(torch_result)} boxes")
    except ImportError:
        print("torchvision.ops.nms not available for comparison")
    
    return boxes_list, scores_list

def benchmark_mojo_nms():
    """Benchmark Mojo NMS performance."""
    
    print("\nBenchmarking Mojo NMS...")
    
    # Test different problem sizes
    problem_sizes = [100, 500, 1000, 2000, 5000]
    num_trials = 10
    
    for n in problem_sizes:
        print(f"\nTesting {n} boxes:")
        
        # Create test data
        boxes = torch.rand(n, 4)
        scores = torch.rand(n)
        
        # Convert to lists
        boxes_list = boxes.cpu().numpy().flatten().tolist()
        scores_list = scores.cpu().numpy().tolist()
        
        # Simulate Mojo NMS timing
        times = []
        for _ in range(num_trials):
            start_time = time.time()
            # result = mojo_nms_kernel(boxes_list, scores_list, 0.5)
            time.sleep(0.001)  # Simulate execution
            end_time = time.time()
            times.append((end_time - start_time) * 1000)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        print(f"  Average time: {avg_time:.3f} ± {std_time:.3f} ms")

if __name__ == "__main__":
    print("=== Mojo NMS Integration Test ===")
    
    # Test basic integration
    boxes_list, scores_list = test_mojo_nms_integration()
    
    # Run benchmarks
    benchmark_mojo_nms()
    
    print("\n=== Test Complete ===")
    print("Next steps:")
    print("1. Compile the Mojo NMS kernel")
    print("2. Export it as PythonObject")
    print("3. Replace the simulated calls with real Mojo calls")
    print("4. Integrate with PyTorch workflows") 