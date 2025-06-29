#!/usr/bin/env python3
"""
PyTorch integration for Mojo NMS
"""

import torch
import numpy as np
import time
import subprocess
import json
import tempfile
import os
from typing import Tuple, Optional

class MojoNMS:
    """PyTorch-compatible Mojo NMS implementation."""
    
    def __init__(self):
        self.mojo_available = self._check_mojo_availability()
        if not self.mojo_available:
            print("Warning: Mojo not available. Using fallback to PyTorch CPU NMS.")
    
    def _check_mojo_availability(self):
        """Check if Mojo is available in the environment."""
        try:
            result = subprocess.run(['mojo', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def __call__(self, boxes: torch.Tensor, scores: torch.Tensor, 
                 iou_threshold: float = 0.5) -> torch.Tensor:
        """
        PyTorch-compatible NMS function.
        
        Args:
            boxes: torch.Tensor of shape (N, 4) in [x1, y1, x2, y2] format
            scores: torch.Tensor of shape (N,)
            iou_threshold: float, IoU threshold for suppression
            
        Returns:
            torch.Tensor: indices of kept boxes
        """
        return self.nms(boxes, scores, iou_threshold)
    
    def nms(self, boxes: torch.Tensor, scores: torch.Tensor, 
            iou_threshold: float = 0.5) -> torch.Tensor:
        """
        Run NMS using Mojo (with PyTorch CPU fallback).
        
        Args:
            boxes: torch.Tensor of shape (N, 4) in [x1, y1, x2, y2] format
            scores: torch.Tensor of shape (N,)
            iou_threshold: float, IoU threshold for suppression
            
        Returns:
            torch.Tensor: indices of kept boxes
        """
        if not self.mojo_available:
            return self._pytorch_fallback(boxes, scores, iou_threshold)
        
        # Ensure tensors are on CPU and contiguous
        boxes = boxes.cpu().contiguous()
        scores = scores.cpu().contiguous()
        
        # Convert to numpy and flatten
        boxes_np = boxes.numpy()
        scores_np = scores.numpy()
        
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
            # Call Mojo NMS
            result_indices = self._call_mojo_nms(temp_file)
            
            # Convert back to torch tensor
            if result_indices:
                return torch.tensor(result_indices, dtype=torch.long, device=boxes.device)
            else:
                return torch.empty(0, dtype=torch.long, device=boxes.device)
                
        finally:
            # Clean up temporary file
            os.unlink(temp_file)
    
    def _call_mojo_nms(self, data_file: str) -> list:
        """
        Call Mojo NMS function.
        This is where we would integrate with the actual Mojo NMS kernel.
        """
        # For now, simulate the call
        # In the real implementation, this would call the Mojo function
        print(f"Calling Mojo NMS with data from {data_file}")
        
        # Load the data to simulate processing
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        # Simulate NMS logic (keep first few boxes)
        # In reality, this would call the Mojo NMS kernel
        num_boxes = len(data['scores'])
        return list(range(min(3, num_boxes)))  # Simulated result
    
    def _pytorch_fallback(self, boxes: torch.Tensor, scores: torch.Tensor, 
                         iou_threshold: float) -> torch.Tensor:
        """Fallback to PyTorch CPU NMS if Mojo is not available."""
        try:
            from torchvision.ops import nms as torch_nms
            return torch_nms(boxes, scores, iou_threshold)
        except ImportError:
            # If torchvision is not available, implement simple CPU NMS
            return self._simple_cpu_nms(boxes, scores, iou_threshold)
    
    def _simple_cpu_nms(self, boxes: torch.Tensor, scores: torch.Tensor, 
                       iou_threshold: float) -> torch.Tensor:
        """Simple CPU NMS implementation as fallback."""
        if len(boxes) == 0:
            return torch.empty(0, dtype=torch.long)
        
        # Sort by scores (descending)
        _, order = scores.sort(descending=True)
        keep = []
        
        while order.numel() > 0:
            if order.numel() == 1:
                keep.append(order.item())
                break
            
            i = order[0]
            keep.append(i)
            
            # Calculate IoU with remaining boxes
            xx1 = boxes[order[1:], 0].clamp(min=boxes[i, 0])
            yy1 = boxes[order[1:], 1].clamp(min=boxes[i, 1])
            xx2 = boxes[order[1:], 2].clamp(max=boxes[i, 2])
            yy2 = boxes[order[1:], 3].clamp(max=boxes[i, 3])
            
            w = (xx2 - xx1).clamp(min=0)
            h = (yy2 - yy1).clamp(min=0)
            inter = w * h
            
            area1 = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area2 = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
            union = area1 + area2 - inter
            
            iou = inter / union
            idx = (iou <= iou_threshold).nonzero().squeeze()
            if idx.numel() == 0:
                break
            order = order[idx + 1]
        
        return torch.tensor(keep, dtype=torch.long)
    
    def benchmark(self, num_boxes: int = 1000, num_trials: int = 10, 
                 iou_threshold: float = 0.5) -> dict:
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

# Convenience function for direct use
def mojo_nms(boxes: torch.Tensor, scores: torch.Tensor, 
             iou_threshold: float = 0.5) -> torch.Tensor:
    """
    Convenience function for Mojo NMS.
    
    Args:
        boxes: torch.Tensor of shape (N, 4) in [x1, y1, x2, y2] format
        scores: torch.Tensor of shape (N,)
        iou_threshold: float, IoU threshold for suppression
        
    Returns:
        torch.Tensor: indices of kept boxes
    """
    nms_fn = MojoNMS()
    return nms_fn(boxes, scores, iou_threshold)

def test_pytorch_integration():
    """Test PyTorch integration."""
    
    print("=== Testing PyTorch-Mojo NMS Integration ===")
    
    # Create NMS function
    nms_fn = MojoNMS()
    
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
    
    result = nms_fn(boxes, scores, 0.5)
    print("NMS result indices:", result)
    print("Kept boxes:", boxes[result])
    
    # Test with GPU tensors (if available)
    if torch.cuda.is_available():
        print("\n--- GPU Tensor Test ---")
        boxes_gpu = boxes.cuda()
        scores_gpu = scores.cuda()
        
        result_gpu = nms_fn(boxes_gpu, scores_gpu, 0.5)
        print("GPU NMS result indices:", result_gpu)
        print("Result device:", result_gpu.device)
    
    # Benchmark
    print("\n--- Benchmark Test ---")
    benchmark_result = nms_fn.benchmark(1000, 5)
    print("Benchmark result:", benchmark_result)
    
    # Test convenience function
    print("\n--- Convenience Function Test ---")
    result_conv = mojo_nms(boxes, scores, 0.5)
    print("Convenience function result:", result_conv)
    
    print("\n=== Integration Test Complete ===")

if __name__ == "__main__":
    test_pytorch_integration() 