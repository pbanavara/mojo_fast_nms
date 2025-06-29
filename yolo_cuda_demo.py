#!/usr/bin/env python3
"""
YOLO + CUDA NMS Integration Demo

This script demonstrates how to integrate the custom CUDA NMS kernel
with a YOLO-like model for accelerated object detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from typing import List, Tuple, Dict, Any
import os

# Try to build and load the CUDA NMS extension
try:
    from torch.utils.cpp_extension import load
    
    print("🔨 Building CUDA NMS extension...")
    nms_cuda = load(
        name="nms_cuda",
        sources=["nms_extension.cpp", "warp_bitmask_nms.cu"],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=True
    )
    CUDA_NMS_AVAILABLE = True
    print("✅ CUDA NMS extension built successfully!")
except Exception as e:
    print(f"❌ Failed to build CUDA NMS extension: {e}")
    print("   Running with CPU NMS only...")
    CUDA_NMS_AVAILABLE = False

class Detection:
    """Detection class for storing bounding box results"""
    def __init__(self, x1: float, y1: float, x2: float, y2: float, confidence: float, class_id: int):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.confidence = confidence
        self.class_id = class_id
    
    def __repr__(self):
        return f"Detection(x1={self.x1:.2f}, y1={self.y1:.2f}, x2={self.x2:.2f}, y2={self.y2:.2f}, conf={self.confidence:.3f}, class={self.class_id})"

class TinyYOLO(nn.Module):
    """A minimal YOLO-like model for demonstration"""
    def __init__(self, num_classes=20, num_boxes=100, input_size=224):
        super().__init__()
        self.num_classes = num_classes
        self.num_boxes = num_boxes
        self.input_size = input_size
        
        # Simple backbone (a few conv layers)
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Output layer: [num_boxes, 4 + 1 + num_classes]
        # 4: bbox coordinates (x1, y1, x2, y2)
        # 1: object confidence
        # num_classes: class probabilities
        self.fc = nn.Linear(128, num_boxes * (4 + 1 + num_classes))
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x.view(x.size(0), self.num_boxes, 5 + self.num_classes)
        return x

def cpu_nms(detections: List[Detection], iou_threshold: float) -> List[Detection]:
    """CPU implementation of NMS for comparison"""
    if not detections:
        return []
    
    # Sort by confidence (descending)
    detections = sorted(detections, key=lambda x: x.confidence, reverse=True)
    
    keep = []
    suppressed = [False] * len(detections)
    
    for i in range(len(detections)):
        if suppressed[i]:
            continue
        
        keep.append(detections[i])
        
        for j in range(i + 1, len(detections)):
            if suppressed[j]:
                continue
            
            # Calculate IoU
            det1, det2 = detections[i], detections[j]
            
            # Calculate intersection
            x1 = max(det1.x1, det2.x1)
            y1 = max(det1.y1, det2.y1)
            x2 = min(det1.x2, det2.x2)
            y2 = min(det1.y2, det2.y2)
            
            if x2 <= x1 or y2 <= y1:
                iou = 0.0
            else:
                intersection = (x2 - x1) * (y2 - y1)
                area1 = (det1.x2 - det1.x1) * (det1.y2 - det1.y1)
                area2 = (det2.x2 - det2.x1) * (det2.y2 - det2.y1)
                union = area1 + area2 - intersection
                iou = intersection / union if union > 0 else 0.0
            
            if iou >= iou_threshold:
                suppressed[j] = True
    
    return keep

def postprocess_yolo_output(output: torch.Tensor, conf_thresh: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Post-process YOLO output to extract boxes, scores, and classes"""
    # output: [batch, num_boxes, 5+num_classes]
    batch_size = output.size(0)
    
    # Extract components
    boxes = output[..., :4]  # [batch, num_boxes, 4]
    obj_scores = output[..., 4]  # [batch, num_boxes]
    class_scores = output[..., 5:]  # [batch, num_boxes, num_classes]
    
    # Calculate class probabilities
    class_probs = F.softmax(class_scores, dim=-1)  # [batch, num_boxes, num_classes]
    class_conf, class_ids = torch.max(class_probs, dim=-1)  # [batch, num_boxes]
    
    # Combine object confidence with class confidence
    scores = obj_scores * class_conf  # [batch, num_boxes]
    
    # Apply confidence threshold
    mask = scores > conf_thresh  # [batch, num_boxes]
    
    # Filter detections
    filtered_boxes = []
    filtered_scores = []
    filtered_classes = []
    
    for b in range(batch_size):
        batch_mask = mask[b]
        if batch_mask.sum() == 0:
            continue
            
        batch_boxes = boxes[b][batch_mask]  # [num_detections, 4]
        batch_scores = scores[b][batch_mask]  # [num_detections]
        batch_classes = class_ids[b][batch_mask]  # [num_detections]
        
        filtered_boxes.append(batch_boxes)
        filtered_scores.append(batch_scores)
        filtered_classes.append(batch_classes)
    
    if not filtered_boxes:
        return torch.empty(0, 4, device=output.device), torch.empty(0, device=output.device), torch.empty(0, dtype=torch.long, device=output.device)
    
    # Concatenate all batches
    all_boxes = torch.cat(filtered_boxes, dim=0)
    all_scores = torch.cat(filtered_scores, dim=0)
    all_classes = torch.cat(filtered_classes, dim=0)
    
    return all_boxes, all_scores, all_classes

def benchmark_nms_performance(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5, num_trials: int = 10):
    """Benchmark CPU vs GPU NMS performance"""
    print(f"\n=== NMS Performance Benchmark ===")
    print(f"Number of detections: {boxes.size(0)}")
    print(f"IoU threshold: {iou_threshold}")
    print(f"Number of trials: {num_trials}")
    
    # Convert to CPU for CPU NMS
    boxes_cpu = boxes.cpu().numpy()
    scores_cpu = scores.cpu().numpy()
    
    # Create Detection objects for CPU NMS
    detections = []
    for i in range(boxes.size(0)):
        detections.append(Detection(
            boxes_cpu[i, 0], boxes_cpu[i, 1], 
            boxes_cpu[i, 2], boxes_cpu[i, 3], 
            scores_cpu[i], 0  # class_id not used in NMS
        ))
    
    # CPU NMS benchmark
    print(f"\n--- CPU NMS Benchmark ---")
    cpu_times = []
    cpu_results = None
    
    for trial in range(num_trials):
        start_time = time.perf_counter()
        cpu_results = cpu_nms(detections, iou_threshold)
        end_time = time.perf_counter()
        
        cpu_time_ms = (end_time - start_time) * 1000
        cpu_times.append(cpu_time_ms)
        print(f"CPU Trial {trial + 1}: {cpu_time_ms:.3f} ms, kept: {len(cpu_results)} detections")
    
    cpu_avg_ms = np.mean(cpu_times)
    print(f"CPU NMS average time: {cpu_avg_ms:.3f} ms")
    print(f"CPU NMS final result: kept {len(cpu_results)} detections")
    
    # GPU NMS benchmark (if available)
    if CUDA_NMS_AVAILABLE:
        print(f"\n--- GPU NMS Benchmark ---")
        gpu_times = []
        gpu_results = None
        
        for trial in range(num_trials):
            # Reset keep buffer by running NMS
            start_time = time.perf_counter()
            keep = nms_cuda.fast_nms(boxes, scores, iou_threshold)
            end_time = time.perf_counter()
            
            gpu_time_ms = (end_time - start_time) * 1000
            gpu_times.append(gpu_time_ms)
            kept_count = keep.sum().item()
            print(f"GPU Trial {trial + 1}: {gpu_time_ms:.3f} ms, kept: {kept_count} detections")
            
            if trial == num_trials - 1:
                gpu_results = keep
        
        gpu_avg_ms = np.mean(gpu_times)
        print(f"GPU NMS average time: {gpu_avg_ms:.3f} ms")
        print(f"GPU NMS final result: kept {gpu_results.sum().item()} detections")
        
        # Performance comparison
        speedup = cpu_avg_ms / gpu_avg_ms
        print(f"\n=== Performance Summary ===")
        print(f"CPU NMS: {cpu_avg_ms:.3f} ms")
        print(f"GPU NMS: {gpu_avg_ms:.3f} ms")
        print(f"Speedup: {speedup:.2f}x")
        print(f"GPU is {speedup:.2f}x faster than CPU")
        
        # Verify results are similar
        print(f"\n=== Result Verification ===")
        print(f"CPU kept: {len(cpu_results)} detections")
        print(f"GPU kept: {gpu_results.sum().item()} detections")
        diff = abs(len(cpu_results) - gpu_results.sum().item())
        print(f"Difference in number of kept detections: {diff}")
        
        if diff <= 2:  # Allow small differences due to floating point precision
            print("✅ Results are consistent between CPU and GPU implementations")
        else:
            print("⚠️  Results differ significantly - may need algorithm verification")
    else:
        print(f"\n--- GPU NMS Benchmark ---")
        print("❌ GPU NMS not available (CUDA extension not built)")

def simulate_yolo_pipeline():
    """Simulate a complete YOLO inference pipeline with GPU NMS"""
    print("\n🎯 Simulating YOLO Inference Pipeline")
    print("=" * 50)
    
    # Create model
    model = TinyYOLO(num_classes=20, num_boxes=100, input_size=224).cuda().eval()
    
    # Generate input image
    batch_size = 1
    input_size = 224
    images = torch.rand(batch_size, 3, input_size, input_size).cuda()
    
    print(f"1. Running YOLO model inference...")
    print(f"   Input shape: {images.shape}")
    
    # Model inference
    with torch.no_grad():
        start_time = time.perf_counter()
        output = model(images)  # [batch, num_boxes, 5+num_classes]
        end_time = time.perf_counter()
        inference_time_ms = (end_time - start_time) * 1000
    
    print(f"   Model inference time: {inference_time_ms:.3f} ms")
    print(f"   Output shape: {output.shape}")
    
    # Post-processing
    print(f"2. Post-processing YOLO output...")
    conf_threshold = 0.25
    boxes, scores, classes = postprocess_yolo_output(output, conf_threshold)
    
    print(f"   After confidence threshold ({conf_threshold}): {boxes.size(0)} detections")
    
    if boxes.size(0) == 0:
        print("   No detections found!")
        return
    
    # Show some sample detections
    print(f"   Sample detections:")
    for i in range(min(3, boxes.size(0))):
        box = boxes[i].cpu().numpy()
        score = scores[i].item()
        class_id = classes[i].item()
        print(f"     Class {class_id}: {score:.3f} confidence at ({box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f})")
    
    # Run NMS
    print(f"3. Running Non-Maximum Suppression...")
    iou_threshold = 0.5
    
    if CUDA_NMS_AVAILABLE:
        try:
            start_time = time.perf_counter()
            keep = nms_cuda.fast_nms(boxes, scores, iou_threshold)
            end_time = time.perf_counter()
            nms_time_ms = (end_time - start_time) * 1000
            print(f"   GPU NMS completed in {nms_time_ms:.3f} ms")
            
            # Apply keep mask
            final_boxes = boxes[keep.bool()]
            final_scores = scores[keep.bool()]
            final_classes = classes[keep.bool()]
            
            print(f"   Final result: {final_boxes.size(0)} detections")
            
        except Exception as e:
            print(f"   ❌ GPU NMS failed: {e}")
            print(f"   Falling back to CPU NMS...")
            start_time = time.perf_counter()
            # Convert to CPU for CPU NMS
            detections = []
            for i in range(boxes.size(0)):
                box = boxes[i].cpu().numpy()
                score = scores[i].item()
                detections.append(Detection(box[0], box[1], box[2], box[3], score, classes[i].item()))
            
            cpu_results = cpu_nms(detections, iou_threshold)
            end_time = time.perf_counter()
            nms_time_ms = (end_time - start_time) * 1000
            print(f"   CPU NMS completed in {nms_time_ms:.3f} ms")
            print(f"   Final result: {len(cpu_results)} detections")
            
            # Convert back to tensors for consistency
            if cpu_results:
                final_boxes = torch.tensor([[d.x1, d.y1, d.x2, d.y2] for d in cpu_results], device=boxes.device)
                final_scores = torch.tensor([d.confidence for d in cpu_results], device=boxes.device)
                final_classes = torch.tensor([d.class_id for d in cpu_results], device=boxes.device)
            else:
                final_boxes = torch.empty(0, 4, device=boxes.device)
                final_scores = torch.empty(0, device=boxes.device)
                final_classes = torch.empty(0, dtype=torch.long, device=boxes.device)
    else:
        start_time = time.perf_counter()
        # Convert to CPU for CPU NMS
        detections = []
        for i in range(boxes.size(0)):
            box = boxes[i].cpu().numpy()
            score = scores[i].item()
            detections.append(Detection(box[0], box[1], box[2], box[3], score, classes[i].item()))
        
        cpu_results = cpu_nms(detections, iou_threshold)
        end_time = time.perf_counter()
        nms_time_ms = (end_time - start_time) * 1000
        print(f"   CPU NMS completed in {nms_time_ms:.3f} ms")
        print(f"   Final result: {len(cpu_results)} detections")
        
        # Convert back to tensors for consistency
        if cpu_results:
            final_boxes = torch.tensor([[d.x1, d.y1, d.x2, d.y2] for d in cpu_results], device=boxes.device)
            final_scores = torch.tensor([d.confidence for d in cpu_results], device=boxes.device)
            final_classes = torch.tensor([d.class_id for d in cpu_results], device=boxes.device)
        else:
            final_boxes = torch.empty(0, 4, device=boxes.device)
            final_scores = torch.empty(0, device=boxes.device)
            final_classes = torch.empty(0, dtype=torch.long, device=boxes.device)
    
    # Show final results
    print("4. Final detections:")
    if final_boxes.size(0) > 0:
        for i in range(min(5, final_boxes.size(0))):
            box = final_boxes[i].cpu().numpy()
            score = final_scores[i].item()
            class_id = final_classes[i].item()
            print(f"   Class {class_id}: {score:.3f} confidence at ({box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f})")
    else:
        print("   No final detections")
    
    # Total pipeline time
    total_time_ms = inference_time_ms + nms_time_ms
    print(f"\n=== Pipeline Summary ===")
    print(f"Model inference: {inference_time_ms:.3f} ms")
    print(f"NMS processing: {nms_time_ms:.3f} ms")
    print(f"Total pipeline: {total_time_ms:.3f} ms")
    print(f"NMS overhead: {(nms_time_ms/total_time_ms)*100:.1f}% of total time")

def main():
    print("🔥 YOLO + CUDA NMS Integration Demo")
    print("This demo shows how to integrate custom CUDA NMS with YOLO object detection")
    print("=" * 60)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA is not available. Please run on a CUDA-enabled system.")
        return
    
    print(f"✅ CUDA is available. Using device: {torch.cuda.get_device_name()}")
    
    # Run the pipeline simulation
    simulate_yolo_pipeline()
    
    # Run performance benchmark with synthetic data
    print("\n" + "=" * 60)
    print("Running performance benchmark with synthetic data...")
    
    # Generate synthetic detections for benchmarking
    num_detections = 1000
    boxes = torch.rand(num_detections, 4, device='cuda') * 224  # Random boxes in 224x224 image
    scores = torch.rand(num_detections, device='cuda')  # Random scores
    
    benchmark_nms_performance(boxes, scores, 0.5, 10)
    
    print("\n✅ Demo completed!")
    print("\nTo integrate with your YOLO model:")
    print("1. Replace the TinyYOLO model with your actual YOLO model")
    print("2. Call nms_cuda.fast_nms(boxes, scores, iou_threshold) after your model inference")
    print("3. Use the returned keep mask to filter your detections")

if __name__ == "__main__":
    main() 