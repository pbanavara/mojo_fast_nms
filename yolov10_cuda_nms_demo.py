import torch
from ultralytics import YOLO
import cv2
import numpy as np
import os
import sys

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
boxes = boxes.cuda()
scores = scores.cuda()

# Run your CUDA NMS
iou_threshold = 0.5
print(f"Running CUDA NMS with IoU threshold {iou_threshold}...")
keep = nms_cuda.fast_nms(boxes, scores, iou_threshold)
final_boxes = boxes[keep.bool()]
final_scores = scores[keep.bool()]
final_classes = classes[keep.bool()]

print(f"Kept {final_boxes.shape[0]} boxes after CUDA NMS")

# Optionally, draw results
for box, score, cls in zip(final_boxes.cpu().numpy(), final_scores.cpu().numpy(), final_classes.cpu().numpy()):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
    cv2.putText(img, f"{int(cls)}:{score:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

cv2.imwrite('yolov10_cuda_nms_result.jpg', img)
print("Result saved to yolov10_cuda_nms_result.jpg") 