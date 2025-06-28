import os
import cv2
import numpy as np
import json
from ultralytics import YOLO

# === CONFIG ===
MODEL_PATH = "runs/detect/train_11l/weights/best.pt"  # ← your trained YOLO model
RGB_IMAGE_PATH = "test.jpg"      # ← RGB test image
DEPTH_IMAGE_PATH = "test.png"    # ← Depth image (same name, different extension)
OUTPUT_DIR = "results"           # ← Where to save cropped RGB and depth images
INTRINSICS_PATH = "intrinsics/camera_intrinsics.json"  # ← Camera intrinsics file for pose estimation
CROP_METADATA_PATH = os.path.join(OUTPUT_DIR, "crop_metadata.json")

# === SETUP ===
os.makedirs(OUTPUT_DIR, exist_ok=True)
rgb_img = cv2.imread(RGB_IMAGE_PATH)
depth_img = cv2.imread(DEPTH_IMAGE_PATH, cv2.IMREAD_UNCHANGED)

assert rgb_img is not None, "❌ RGB image not found or unreadable"
assert depth_img is not None, "❌ Depth image not found or unreadable"
assert rgb_img.shape[:2] == depth_img.shape[:2], "❌ RGB and depth resolution mismatch"

# === Load YOLO Model and Run Inference ===
model = YOLO(MODEL_PATH)
results = model(rgb_img, verbose=False)
detections = results[0].boxes
labels = model.names

# === Crop & Save ===
metadata = []

for i, box in enumerate(detections):
    cls_id = int(box.cls.item())
    conf = box.conf.item()
    if conf < 0.5:
        continue

    class_name = labels[cls_id]
    xyxy = box.xyxy.cpu().numpy().squeeze()
    xmin, ymin, xmax, ymax = map(int, xyxy)

    # Clip coordinates to image boundaries
    h, w = rgb_img.shape[:2]
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(w, xmax)
    ymax = min(h, ymax)

    # Crop RGB and Depth
    crop_rgb = rgb_img[ymin:ymax, xmin:xmax]
    crop_depth = depth_img[ymin:ymax, xmin:xmax]

    # Save cropped images
    rgb_crop_filename = f"rgb_crop_{i}_{class_name}.png"
    depth_crop_filename = f"depth_crop_{i}_{class_name}.png"

    cv2.imwrite(os.path.join(OUTPUT_DIR, rgb_crop_filename), crop_rgb)
    cv2.imwrite(os.path.join(OUTPUT_DIR, depth_crop_filename), crop_depth)

    metadata.append({
        "id": i,
        "class": class_name,
        "rgb": rgb_crop_filename,
        "depth": depth_crop_filename,
        "bbox": [xmin, ymin, xmax, ymax],
        "conf": conf
    })

    print(f"[{i}] Saved: {rgb_crop_filename}, {depth_crop_filename}")

# Save metadata for pose overlay later
with open(CROP_METADATA_PATH, 'w') as f:
    json.dump(metadata, f, indent=2)

print("✅ Cropping completed and saved in 'results/'")
