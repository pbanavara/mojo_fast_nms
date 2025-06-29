import torch
from ultralytics import YOLO
import cv2
import numpy as np
import os
import sys
import time

# Try to load the existing CUDA extension
try:
    # Add the existing extension path to sys.path
    extension_path = "/home/ubuntu/.cache/torch_extensions/py310_cu128/nms_cuda"
    if extension_path not in sys.path:
        sys.path.insert(0, extension_path)
    
    import nms_cuda
    print("✅ Loaded existing CUDA NMS extension")
except ImportError:
    print("❌ Could not load existing CUDA NMS extension")
    print("   Please run the previous demo first to build the extension")
    sys.exit(1)

class Detection:
    def __init__(self, x1, y1, x2, y2, confidence, class_id):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.confidence = confidence
        self.class_id = class_id

    def __repr__(self):
        return f"Detection(x1={self.x1:.2f}, y1={self.y1:.2f}, x2={self.x2:.2f}, y2={self.y2:.2f}, conf={self.confidence:.3f}, class={self.class_id})"

def cpu_nms(detections, iou_threshold):
    if not detections:
        return []
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
            det1, det2 = detections[i], detections[j]
            x1 = max(det1.x1, det2.x1)
            y1 = max(det1.y1, det2.y1)
            x2 = min(det1.x2, det2.x2)
            y2 = min(det1.y2, det2.y2)
            if x2 <= x1 or y2 <= y1:
                iou = 0.0
            else:
                intersection = (x2 - x1) * (y2 - y1)
                area1 = (det1.x2 - det1.x1) * (det1.y2 - det1.y1)
                area2 = (det2.x2 - det2.y1) * (det2.y2 - det2.y1)
                union = area1 + area2 - intersection
                iou = intersection / union if union > 0 else 0.0
            if iou >= iou_threshold:
                suppressed[j] = True
    return keep

# Load YOLOv10 model
print("Loading YOLOv10 model...")
model = YOLO('yolov10s.pt')  # or yolov10m.pt, yolov10l.pt, etc.

# Load and preprocess image
img_path = 'image.jpg'  # <-- Change to your image path
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Image not found: {img_path}")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print(f"Running YOLOv10 inference on {img_path}...")
# Run YOLOv10 inference
results = model(img_rgb, device=0)  # device=0 for CUDA

# Extract raw boxes, scores, and classes
boxes = results[0].boxes.xyxy  # [N, 4], xyxy format
scores = results[0].boxes.conf  # [N]
classes = results[0].boxes.cls  # [N]

print(f"YOLOv10 detected {len(boxes)} objects before NMS")

# Convert to CUDA tensors
boxes_cuda = boxes.cuda()
scores_cuda = scores.cuda()

# CUDA NMS
iou_threshold = 0.5
start_cuda = time.perf_counter()
keep_cuda = nms_cuda.fast_nms(boxes_cuda, scores_cuda, iou_threshold)
end_cuda = time.perf_counter()
final_boxes_cuda = boxes_cuda[keep_cuda.bool()]
final_scores_cuda = scores_cuda[keep_cuda.bool()]
final_classes_cuda = classes[keep_cuda.bool()]
cuda_time_ms = (end_cuda - start_cuda) * 1000
print(f"CUDA NMS: Kept {final_boxes_cuda.shape[0]} boxes in {cuda_time_ms:.3f} ms")

# Draw CUDA NMS results
img_cuda = img.copy()
for box, score, cls in zip(final_boxes_cuda.cpu().numpy(), final_scores_cuda.cpu().numpy(), final_classes_cuda.cpu().numpy()):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img_cuda, (x1, y1), (x2, y2), (0,255,0), 2)
    cv2.putText(img_cuda, f"{int(cls)}:{score:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
cv2.imwrite('benchmarks_yolov10/yolov10_cuda_nms_result.jpg', img_cuda)

# CPU NMS
boxes_cpu = boxes.cpu().numpy()
scores_cpu = scores.cpu().numpy()
classes_cpu = classes.cpu().numpy()
detections = [Detection(boxes_cpu[i,0], boxes_cpu[i,1], boxes_cpu[i,2], boxes_cpu[i,3], scores_cpu[i], classes_cpu[i]) for i in range(len(boxes_cpu))]
start_cpu = time.perf_counter()
final_cpu = cpu_nms(detections, iou_threshold)
end_cpu = time.perf_counter()
cpu_time_ms = (end_cpu - start_cpu) * 1000
print(f"CPU NMS: Kept {len(final_cpu)} boxes in {cpu_time_ms:.3f} ms")

# Draw CPU NMS results
img_cpu = img.copy()
for det in final_cpu:
    x1, y1, x2, y2 = map(int, [det.x1, det.y1, det.x2, det.y2])
    cv2.rectangle(img_cpu, (x1, y1), (x2, y2), (255,0,0), 2)
    cv2.putText(img_cpu, f"{int(det.class_id)}:{det.confidence:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
cv2.imwrite('benchmarks_yolov10/yolov10_cpu_nms_result.jpg', img_cpu)

print(f"\n=== YOLOv10 NMS Benchmark ===")
print(f"CUDA NMS: {final_boxes_cuda.shape[0]} boxes, {cuda_time_ms:.3f} ms (green boxes)")
print(f"CPU NMS:  {len(final_cpu)} boxes, {cpu_time_ms:.3f} ms (blue boxes)")
print("Results saved to benchmarks_yolov10/yolov10_cuda_nms_result.jpg and yolov10_cpu_nms_result.jpg") 