#!/usr/bin/env python3
"""
Python wrapper for Mojo NMS integration with PyTorch
"""

import numpy as np
import torch
import time
import subprocess
import json
import tempfile
import os

class MojoNMSWrapper:
    """Wrapper for Mojo NMS functions."""
    
    def __init__(self):
        self.mojo_available = self._check_mojo_availability()
        
    def _check_mojo_availability(self):
        """Check if Mojo is available in the environment."""
        try:
            result = subprocess.run(['mojo', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def nms(self, boxes, scores, iou_threshold=0.5):
        """
        Run NMS using Mojo.
        
        Args:
            boxes: torch.Tensor of shape (N, 4) in [x1, y1, x2, y2] format
            scores: torch.Tensor of shape (N,)
            iou_threshold: float, IoU threshold for suppression
            
        Returns:
            torch.Tensor: indices of kept boxes
        """
        if not self.mojo_available:
            raise RuntimeError("Mojo not available. Please activate pixi shell.")
        
        # Convert to numpy and flatten
        boxes_np = boxes.cpu().numpy()
        scores_np = scores.cpu().numpy()
        
        # Convert to YOLO format [y1, x1, y2, x2]
        boxes_yolo = boxes_np[:, [1, 0, 3, 2]]  # Swap x and y coordinates
        
        # Flatten boxes
        boxes_flat = boxes_yolo.flatten().tolist()
        scores_list = scores_np.tolist()
        
        # Create temporary file with data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            data = {
                'boxes': boxes_flat,
                'scores': scores_list,
                'iou_threshold': iou_threshold
            }
            json.dump(data, f)
            temp_file = f.name
        
        try:
            # Call Mojo NMS (this would be the actual implementation)
            # For now, we'll simulate the call
            result_indices = self._call_mojo_nms(temp_file)
            
            # Convert back to torch tensor
            if result_indices:
                return torch.tensor(result_indices, dtype=torch.long)
            else:
                return torch.empty(0, dtype=torch.long)
                
        finally:
            # Clean up temporary file
            os.unlink(temp_file)
    
    def _call_mojo_nms(self, data_file):
        """
        Call Mojo NMS function.
        This is where we would integrate with the actual Mojo NMS kernel.
        """
        # For now, simulate the call
        # In the real implementation, this would call the Mojo function
        print(f"Would call Mojo NMS with data from {data_file}")
        
        # Simulate some NMS logic (keep first few boxes)
        return [0, 2, 4]  # Simulated result
    
    def benchmark(self, num_boxes=1000, num_trials=10, iou_threshold=0.5):
        """Benchmark NMS performance."""
        
        print(f"Benchmarking Mojo NMS with {num_boxes} boxes, {num_trials} trials...")
        
        # Generate test data
        boxes = torch.rand(num_boxes, 4)
        scores = torch.rand(num_boxes)
        
        # Sort by scores
        sorted_indices = torch.argsort(scores, descending=True)
        boxes = boxes[sorted_indices]
        scores = scores[sorted_indices]
        
        # Warm up
        _ = self.nms(boxes[:100], scores[:100], iou_threshold)
        
        # Benchmark
        times = []
        results = []
        
        for i in range(num_trials):
            start_time = time.time()
            result = self.nms(boxes, scores, iou_threshold)
            end_time = time.time()
            
            times.append((end_time - start_time) * 1000)  # Convert to ms
            results.append(len(result))
            
            if i == 0:
                print(f"First run: kept {len(result)} boxes")
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        avg_kept = np.mean(results)
        
        print(f"Average time: {avg_time:.3f} ± {std_time:.3f} ms")
        print(f"Average boxes kept: {avg_kept:.1f}")
        
        return {
            'avg_time_ms': avg_time,
            'std_time_ms': std_time,
            'avg_kept': avg_kept,
            'num_boxes': num_boxes,
            'num_trials': num_trials
        }

def test_mojo_nms_integration():
    """Test the Mojo NMS integration."""
    
    print("=== Testing Mojo NMS Integration ===")
    
    # Create wrapper
    nms_wrapper = MojoNMSWrapper()
    
    if not nms_wrapper.mojo_available:
        print("❌ Mojo not available. Please activate pixi shell.")
        return
    
    print("✅ Mojo is available")
    
    # Test basic NMS
    print("\n--- Basic NMS Test ---")
    boxes = torch.tensor([
        [0, 0, 10, 10],
        [1, 1, 11, 11],
        [5, 5, 15, 15],
        [20, 20, 30, 30]
    ], dtype=torch.float32)
    
    scores = torch.tensor([0.9, 0.8, 0.7, 0.6], dtype=torch.float32)
    
    print("Input boxes:", boxes)
    print("Input scores:", scores)
    
    result = nms_wrapper.nms(boxes, scores, 0.5)
    print("NMS result indices:", result)
    
    # Benchmark
    print("\n--- Benchmark Test ---")
    benchmark_result = nms_wrapper.benchmark(1000, 5)
    print("Benchmark result:", benchmark_result)
    
    # Compare with PyTorch CPU NMS
    try:
        from torchvision.ops import nms as torch_nms
        print("\n--- PyTorch CPU NMS Comparison ---")
        
        start_time = time.time()
        torch_result = torch_nms(boxes, scores, 0.5)
        torch_time = (time.time() - start_time) * 1000
        
        print(f"PyTorch CPU NMS: {torch_time:.3f} ms, kept {len(torch_result)} boxes")
        print(f"PyTorch result indices: {torch_result}")
        
    except ImportError:
        print("torchvision.ops.nms not available for comparison")

if __name__ == "__main__":
    test_mojo_nms_integration() 