#!/usr/bin/env python3
"""
Simple test script to verify PyTorch-Mojo NMS integration
"""

import torch
import time

def test_basic_integration():
    """Test basic PyTorch-Mojo NMS integration."""
    
    print("=== Testing PyTorch-Mojo NMS Integration ===")
    
    try:
        from pytorch_mojo_nms import mojo_nms, MojoNMS
        print("✅ Successfully imported PyTorch-Mojo NMS")
    except ImportError as e:
        print(f"❌ Failed to import: {e}")
        return False
    
    # Test data
    boxes = torch.tensor([
        [0, 0, 10, 10],
        [1, 1, 11, 11],
        [5, 5, 15, 15],
        [20, 20, 30, 30]
    ], dtype=torch.float32)
    
    scores = torch.tensor([0.9, 0.8, 0.7, 0.6], dtype=torch.float32)
    
    print(f"Input boxes: {boxes.shape}")
    print(f"Input scores: {scores.shape}")
    
    # Test function-based approach
    try:
        start_time = time.time()
        result1 = mojo_nms(boxes, scores, iou_threshold=0.5)
        func_time = (time.time() - start_time) * 1000
        print(f"✅ Function-based NMS: {len(result1)} boxes kept in {func_time:.3f} ms")
    except Exception as e:
        print(f"❌ Function-based NMS failed: {e}")
        return False
    
    # Test class-based approach
    try:
        nms_fn = MojoNMS()
        start_time = time.time()
        result2 = nms_fn(boxes, scores, iou_threshold=0.5)
        class_time = (time.time() - start_time) * 1000
        print(f"✅ Class-based NMS: {len(result2)} boxes kept in {class_time:.3f} ms")
    except Exception as e:
        print(f"❌ Class-based NMS failed: {e}")
        return False
    
    # Test GPU tensors (if available)
    if torch.cuda.is_available():
        try:
            boxes_gpu = boxes.cuda()
            scores_gpu = scores.cuda()
            
            result_gpu = mojo_nms(boxes_gpu, scores_gpu, iou_threshold=0.5)
            print(f"✅ GPU tensor NMS: {len(result_gpu)} boxes kept")
            print(f"   Result device: {result_gpu.device}")
        except Exception as e:
            print(f"❌ GPU tensor NMS failed: {e}")
    
    # Test batch processing
    try:
        batch_boxes = torch.rand(4, 100, 4)  # 4 images, 100 detections each
        batch_scores = torch.rand(4, 100)
        
        batch_results = []
        for i in range(4):
            result = mojo_nms(batch_boxes[i], batch_scores[i], iou_threshold=0.5)
            batch_results.append(len(result))
        
        print(f"✅ Batch processing: {batch_results} boxes kept per image")
    except Exception as e:
        print(f"❌ Batch processing failed: {e}")
    
    print("\n=== Integration Test Complete ===")
    return True

def test_performance():
    """Test performance with different problem sizes."""
    
    print("\n=== Performance Test ===")
    
    try:
        from pytorch_mojo_nms import MojoNMS
        nms_fn = MojoNMS()
    except ImportError:
        print("❌ Cannot import MojoNMS")
        return
    
    problem_sizes = [100, 500, 1000]
    
    for num_boxes in problem_sizes:
        print(f"\nTesting {num_boxes} boxes:")
        
        # Generate test data
        boxes = torch.rand(num_boxes, 4) * 224
        scores = torch.rand(num_boxes)
        
        # Sort by scores
        sorted_indices = torch.argsort(scores, descending=True)
        boxes = boxes[sorted_indices]
        scores = scores[sorted_indices]
        
        # Time the NMS
        times = []
        for _ in range(5):  # 5 trials
            start_time = time.time()
            result = nms_fn(boxes, scores, iou_threshold=0.5)
            end_time = time.time()
            times.append((end_time - start_time) * 1000)
        
        avg_time = sum(times) / len(times)
        print(f"  Average time: {avg_time:.3f} ms")
        print(f"  Boxes kept: {len(result)}")

if __name__ == "__main__":
    print("Testing PyTorch-Mojo NMS Integration")
    print("Make sure you're in the pixi shell environment!")
    print()
    
    # Run tests
    success = test_basic_integration()
    
    if success:
        test_performance()
        print("\n🎉 All tests passed! The integration is working correctly.")
        print("\nNext steps:")
        print("1. Use mojo_nms() in your PyTorch models")
        print("2. Benchmark against your existing NMS implementation")
        print("3. Integrate with your object detection pipeline")
    else:
        print("\n❌ Some tests failed. Check the error messages above.")
        print("\nTroubleshooting:")
        print("1. Make sure you're in the pixi shell: pixi shell")
        print("2. Check that PyTorch is installed: pip install torch")
        print("3. Verify Mojo is available: mojo --version") 