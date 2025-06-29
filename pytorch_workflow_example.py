#!/usr/bin/env python3
"""
Complete PyTorch workflow example using Mojo NMS
"""

import torch
import torch.nn as nn
import numpy as np
import time
from pytorch_mojo_nms import MojoNMS, mojo_nms

class SimpleObjectDetector(nn.Module):
    """Simple object detector that uses Mojo NMS."""
    
    def __init__(self, num_classes=80):
        super().__init__()
        self.num_classes = num_classes
        
        # Simple backbone (just for demonstration)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # Detection heads
        self.bbox_head = nn.Linear(128, 4)  # [x1, y1, x2, y2]
        self.cls_head = nn.Linear(128, num_classes)
        
        # NMS function
        self.nms = MojoNMS()
    
    def forward(self, x):
        """Forward pass with NMS."""
        # Extract features
        features = self.backbone(x)
        
        # Generate detections (simplified)
        batch_size = x.shape[0]
        num_detections = 100  # Fixed number for simplicity
        
        # Generate random detections (in real implementation, this would come from the model)
        boxes = torch.rand(batch_size, num_detections, 4) * 224  # Scale to image size
        scores = torch.rand(batch_size, num_detections)
        
        # Apply NMS to each image in the batch
        results = []
        for i in range(batch_size):
            # Sort by scores
            sorted_indices = torch.argsort(scores[i], descending=True)
            boxes_sorted = boxes[i][sorted_indices]
            scores_sorted = scores[i][sorted_indices]
            
            # Apply NMS
            keep_indices = self.nms(boxes_sorted, scores_sorted, iou_threshold=0.5)
            
            # Get final detections
            final_boxes = boxes_sorted[keep_indices]
            final_scores = scores_sorted[keep_indices]
            
            results.append({
                'boxes': final_boxes,
                'scores': final_scores,
                'num_detections': len(keep_indices)
            })
        
        return results

def benchmark_detector():
    """Benchmark the detector with different batch sizes."""
    
    print("=== Benchmarking Object Detector with Mojo NMS ===")
    
    # Create detector
    detector = SimpleObjectDetector()
    
    # Test different batch sizes
    batch_sizes = [1, 4, 8, 16]
    num_trials = 5
    
    for batch_size in batch_sizes:
        print(f"\n--- Batch Size: {batch_size} ---")
        
        # Create input
        x = torch.rand(batch_size, 3, 224, 224)
        
        # Warm up
        _ = detector(x)
        
        # Benchmark
        times = []
        for _ in range(num_trials):
            start_time = time.time()
            results = detector(x)
            end_time = time.time()
            times.append((end_time - start_time) * 1000)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        # Count total detections
        total_detections = sum(r['num_detections'] for r in results)
        avg_detections = total_detections / batch_size
        
        print(f"Average time: {avg_time:.3f} ± {std_time:.3f} ms")
        print(f"Average detections per image: {avg_detections:.1f}")
        print(f"Throughput: {batch_size / (avg_time / 1000):.1f} images/sec")

def compare_nms_methods():
    """Compare different NMS methods."""
    
    print("\n=== Comparing NMS Methods ===")
    
    # Create test data
    num_boxes = 1000
    boxes = torch.rand(num_boxes, 4) * 224
    scores = torch.rand(num_boxes)
    
    # Sort by scores
    sorted_indices = torch.argsort(scores, descending=True)
    boxes = boxes[sorted_indices]
    scores = scores[sorted_indices]
    
    # Test Mojo NMS
    print("\n--- Mojo NMS ---")
    start_time = time.time()
    mojo_result = mojo_nms(boxes, scores, 0.5)
    mojo_time = (time.time() - start_time) * 1000
    print(f"Time: {mojo_time:.3f} ms")
    print(f"Kept: {len(mojo_result)} boxes")
    
    # Test PyTorch CPU NMS
    try:
        from torchvision.ops import nms as torch_nms
        print("\n--- PyTorch CPU NMS ---")
        start_time = time.time()
        torch_result = torch_nms(boxes, scores, 0.5)
        torch_time = (time.time() - start_time) * 1000
        print(f"Time: {torch_time:.3f} ms")
        print(f"Kept: {len(torch_result)} boxes")
        
        # Compare results
        print(f"\nSpeedup: {torch_time / mojo_time:.2f}x")
        
        # Check if results are similar (they might differ due to different implementations)
        print(f"Result similarity: {len(set(mojo_result.tolist()) & set(torch_result.tolist()))} / {len(mojo_result)}")
        
    except ImportError:
        print("torchvision.ops.nms not available for comparison")

def real_world_example():
    """Real-world example with YOLO-like detections."""
    
    print("\n=== Real-World Example ===")
    
    # Simulate YOLO detections
    num_images = 5
    detections_per_image = 100
    
    for i in range(num_images):
        print(f"\nProcessing image {i+1}/{num_images}")
        
        # Generate realistic detections
        boxes = torch.rand(detections_per_image, 4) * 640  # 640x640 image
        scores = torch.rand(detections_per_image)
        
        # Add some overlapping boxes to test NMS
        for j in range(0, detections_per_image, 10):
            if j + 1 < detections_per_image:
                # Create overlapping boxes
                boxes[j+1] = boxes[j] + torch.rand(4) * 20
        
        # Sort by scores
        sorted_indices = torch.argsort(scores, descending=True)
        boxes = boxes[sorted_indices]
        scores = scores[sorted_indices]
        
        # Apply NMS
        start_time = time.time()
        keep_indices = mojo_nms(boxes, scores, iou_threshold=0.5)
        nms_time = (time.time() - start_time) * 1000
        
        final_boxes = boxes[keep_indices]
        final_scores = scores[keep_indices]
        
        print(f"  Original detections: {len(boxes)}")
        print(f"  After NMS: {len(final_boxes)}")
        print(f"  NMS time: {nms_time:.3f} ms")
        print(f"  Top 3 scores: {final_scores[:3].tolist()}")

def main():
    """Main function to run all examples."""
    
    print("=== PyTorch-Mojo NMS Workflow Examples ===")
    
    # Test basic functionality
    print("\n1. Testing basic detector...")
    detector = SimpleObjectDetector()
    x = torch.rand(1, 3, 224, 224)
    results = detector(x)
    print(f"Detector output: {len(results)} images processed")
    print(f"First image detections: {results[0]['num_detections']}")
    
    # Benchmark detector
    print("\n2. Benchmarking detector...")
    benchmark_detector()
    
    # Compare NMS methods
    print("\n3. Comparing NMS methods...")
    compare_nms_methods()
    
    # Real-world example
    print("\n4. Real-world example...")
    real_world_example()
    
    print("\n=== All Examples Complete ===")
    print("\nUsage in your code:")
    print("from pytorch_mojo_nms import mojo_nms")
    print("keep_indices = mojo_nms(boxes, scores, iou_threshold=0.5)")

if __name__ == "__main__":
    main() 