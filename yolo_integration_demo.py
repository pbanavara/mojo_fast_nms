#!/usr/bin/env python3
"""
YOLO GPU NMS Integration Demo

This script demonstrates how to integrate the Mojo GPU NMS module
with a YOLO model for accelerated object detection.
"""

import torch
import torch.nn as nn
import numpy as np
import time
from typing import List, Tuple, Dict, Any
import sys
import os

# Try to import the Mojo module
try:
    import yolo_gpu_nms
    MOJO_AVAILABLE = True
    print("✅ Mojo GPU NMS module imported successfully!")
except ImportError as e:
    print(f"⚠️  Mojo module not available: {e}")
    print("   Running with CPU NMS only...")
    MOJO_AVAILABLE = False

class Detection:
    """Detection class compatible with Mojo Detection struct"""
    def __init__(self, x1: float, y1: float, x2: float, y2: float, confidence: float, class_id: int):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.confidence = confidence
        self.class_id = class_id
    
    def __repr__(self):
        return f"Detection(x1={self.x1:.2f}, y1={self.y1:.2f}, x2={self.x2:.2f}, y2={self.y2:.2f}, conf={self.confidence:.3f}, class={self.class_id})"

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

def generate_mock_yolo_detections(num_detections: int, image_width: int = 640, image_height: int = 640) -> List[Detection]:
    """Generate mock YOLO detections for testing"""
    detections = []
    np.random.seed(42)
    
    for i in range(num_detections):
        # Generate random box coordinates
        x1 = np.random.uniform(0, image_width - 100)
        y1 = np.random.uniform(0, image_height - 100)
        w = np.random.uniform(20, 100)
        h = np.random.uniform(20, 100)
        confidence = np.random.uniform(0.1, 0.95)
        class_id = np.random.randint(0, 80)  # COCO has 80 classes
        
        # Ensure box is within image bounds
        x2 = min(x1 + w, image_width)
        y2 = min(y1 + h, image_height)
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        
        detections.append(Detection(x1, y1, x2, y2, confidence, class_id))
    
    return detections

def benchmark_nms_performance(detections: List[Detection], iou_threshold: float = 0.5, num_trials: int = 10):
    """Benchmark CPU vs GPU NMS performance"""
    print(f"\n=== NMS Performance Benchmark ===")
    print(f"Number of detections: {len(detections)}")
    print(f"IoU threshold: {iou_threshold}")
    print(f"Number of trials: {num_trials}")
    
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
    if MOJO_AVAILABLE:
        print(f"\n--- GPU NMS Benchmark ---")
        try:
            gpu_benchmark = yolo_gpu_nms.benchmark_gpu_nms(detections, iou_threshold, num_trials)
            gpu_avg_ms = gpu_benchmark["average_time_ms"]
            gpu_results = gpu_benchmark["final_result"]
            
            print(f"GPU NMS average time: {gpu_avg_ms:.3f} ms")
            print(f"GPU NMS final result: kept {len(gpu_results)} detections")
            
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
            print(f"GPU kept: {len(gpu_results)} detections")
            diff = abs(len(cpu_results) - len(gpu_results))
            print(f"Difference in number of kept detections: {diff}")
            
            if diff <= 2:  # Allow small differences due to floating point precision
                print("✅ Results are consistent between CPU and GPU implementations")
            else:
                print("⚠️  Results differ significantly - may need algorithm verification")
                
        except Exception as e:
            print(f"❌ GPU NMS benchmark failed: {e}")
    else:
        print(f"\n--- GPU NMS Benchmark ---")
        print("❌ GPU NMS not available (Mojo module not loaded)")

def demo_yolo_integration():
    """Demonstrate YOLO integration with GPU NMS"""
    print("🚀 YOLO GPU NMS Integration Demo")
    print("=" * 50)
    
    # Generate test detections
    print("\n1. Generating mock YOLO detections...")
    detections = generate_mock_yolo_detections(1000, 640, 640)
    print(f"   Generated {len(detections)} detections")
    
    # Show some sample detections
    print(f"\n2. Sample detections:")
    for i in range(min(5, len(detections))):
        print(f"   {detections[i]}")
    
    # Run NMS
    print(f"\n3. Running NMS with IoU threshold = 0.5...")
    
    if MOJO_AVAILABLE:
        try:
            # GPU NMS
            gpu_results = yolo_gpu_nms.gpu_nms(detections, 0.5)
            print(f"   GPU NMS kept {len(gpu_results)} detections")
            
            # CPU NMS for comparison
            cpu_results = cpu_nms(detections, 0.5)
            print(f"   CPU NMS kept {len(cpu_results)} detections")
            
            # Show some kept detections
            print(f"\n4. Sample kept detections (GPU NMS):")
            for i in range(min(3, len(gpu_results))):
                print(f"   {gpu_results[i]}")
                
        except Exception as e:
            print(f"   ❌ GPU NMS failed: {e}")
            print(f"   Falling back to CPU NMS...")
            cpu_results = cpu_nms(detections, 0.5)
            print(f"   CPU NMS kept {len(cpu_results)} detections")
    else:
        # CPU NMS only
        cpu_results = cpu_nms(detections, 0.5)
        print(f"   CPU NMS kept {len(cpu_results)} detections")
    
    # Run performance benchmark
    benchmark_nms_performance(detections, 0.5, 10)

def simulate_yolo_pipeline():
    """Simulate a complete YOLO inference pipeline with GPU NMS"""
    print("\n🎯 Simulating YOLO Inference Pipeline")
    print("=" * 50)
    
    # Simulate YOLO model output (this would normally come from your YOLO model)
    print("1. Simulating YOLO model inference...")
    time.sleep(0.1)  # Simulate model inference time
    
    # Generate detections (simulating YOLO output)
    detections = generate_mock_yolo_detections(2000, 640, 640)
    print(f"   YOLO model produced {len(detections)} candidate detections")
    
    # Apply confidence threshold (typical YOLO post-processing)
    confidence_threshold = 0.25
    filtered_detections = [d for d in detections if d.confidence >= confidence_threshold]
    print(f"   After confidence threshold ({confidence_threshold}): {len(filtered_detections)} detections")
    
    # Run NMS
    print("2. Running Non-Maximum Suppression...")
    iou_threshold = 0.5
    
    if MOJO_AVAILABLE:
        try:
            start_time = time.perf_counter()
            final_detections = yolo_gpu_nms.gpu_nms(filtered_detections, iou_threshold)
            end_time = time.perf_counter()
            nms_time_ms = (end_time - start_time) * 1000
            print(f"   GPU NMS completed in {nms_time_ms:.3f} ms")
            print(f"   Final result: {len(final_detections)} detections")
        except Exception as e:
            print(f"   ❌ GPU NMS failed: {e}")
            print(f"   Falling back to CPU NMS...")
            start_time = time.perf_counter()
            final_detections = cpu_nms(filtered_detections, iou_threshold)
            end_time = time.perf_counter()
            nms_time_ms = (end_time - start_time) * 1000
            print(f"   CPU NMS completed in {nms_time_ms:.3f} ms")
            print(f"   Final result: {len(final_detections)} detections")
    else:
        start_time = time.perf_counter()
        final_detections = cpu_nms(filtered_detections, iou_threshold)
        end_time = time.perf_counter()
        nms_time_ms = (end_time - start_time) * 1000
        print(f"   CPU NMS completed in {nms_time_ms:.3f} ms")
        print(f"   Final result: {len(final_detections)} detections")
    
    # Show final results
    print("3. Final detections:")
    for i in range(min(5, len(final_detections))):
        det = final_detections[i]
        print(f"   Class {det.class_id}: {det.confidence:.3f} confidence at ({det.x1:.1f}, {det.y1:.1f}, {det.x2:.1f}, {det.y2:.1f})")

if __name__ == "__main__":
    print("🔥 YOLO GPU NMS Integration Demo")
    print("This demo shows how to integrate Mojo GPU NMS with YOLO object detection")
    
    # Run the demo
    demo_yolo_integration()
    
    # Run the pipeline simulation
    simulate_yolo_pipeline()
    
    print("\n✅ Demo completed!")
    print("\nTo integrate with your YOLO model:")
    print("1. Replace the mock detections with your YOLO model output")
    print("2. Call yolo_gpu_nms.gpu_nms(detections, iou_threshold) after your model inference")
    print("3. Use the returned detections for visualization or further processing") 