import os
import glob
import cv2
import numpy as np
from ultralytics import YOLO

# === CONFIG ===
MODEL_PATH = "/workspace/data/YOLO/runs/detect/train_11s7/weights/best.pt"
IMAGE_DIR = "/workspace/data/YOLO/custom_data/images"
OUTPUT_ROOT = "/workspace/data/YOLO/masks"
SCORE_THRESH = 0.75
VISUAL_DEBUG = False  # Set True to see ROI and mask overlays

# Load model
model = YOLO(MODEL_PATH)
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# List all RGB images
img_exts = (".jpg", ".jpeg", ".png", ".bmp")
image_files = sorted([f for f in glob.glob(os.path.join(IMAGE_DIR, "*")) if f.endswith(img_exts)])

def print_mean_rgb(roi, label):
    mean_b = np.mean(roi[:, :, 0])
    mean_g = np.mean(roi[:, :, 1])
    mean_r = np.mean(roi[:, :, 2])
    print(f"[{label}] Mean BGR: ({mean_b:.1f}, {mean_g:.1f}, {mean_r:.1f})")

def print_rgb_stats(roi, label):
    roi_flat = roi.reshape(-1, 3)
    mean_bgr = np.mean(roi_flat, axis=0)
    min_bgr = np.min(roi_flat, axis=0)
    max_bgr = np.max(roi_flat, axis=0)
    print(f"[{label}] Mean BGR: {mean_bgr.round(1)} Min: {min_bgr} Max: {max_bgr}")

def mask_gray_rgb(roi):
    b, g, r = cv2.split(roi)
    mask = (
        (np.abs(r.astype(int) - g.astype(int)) < 22) &
        (np.abs(r.astype(int) - b.astype(int)) < 22) &
        (np.abs(g.astype(int) - b.astype(int)) < 22) &
        (r > 25) & (r < 200) &
        (g > 25) & (g < 200) &
        (b > 25) & (b < 200)
    )
    mask = mask.astype(np.uint8) * 255

    # === Otsu fallback ===
    # If the mask is too small, use Otsu thresholding to catch the shape
    if np.sum(mask) < 100:
        gray_img = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(
            gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    # Morphological filtering (after either mask or Otsu)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Fill holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # Clean small specks/branches
    return mask

def keep_largest_region(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask
    largest = max(contours, key=cv2.contourArea)
    mask_out = np.zeros_like(mask)
    cv2.drawContours(mask_out, [largest], -1, 255, -1)
    return mask_out

def create_mask(img, bbox, label):
    xmin, ymin, xmax, ymax = bbox
    roi = img[ymin:ymax, xmin:xmax]
    if roi.size == 0:
        return None

    # === Step 1: Blur to reduce color noise ===
    roi = cv2.GaussianBlur(roi, (5, 5), 0)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    label_lower = label.lower()

    mask_roi = None

    if "yellow" in label_lower:
        lower = np.array([20, 100, 100])
        upper = np.array([40, 255, 255])
        mask_roi = cv2.inRange(hsv, lower, upper)
        mask_roi = keep_largest_region(mask_roi)  # Keep only largest region

    elif any(x in label_lower for x in ["gray", "grey", "grau", "black"]):
        print_rgb_stats(roi, label)
        mask_roi = mask_gray_rgb(roi)
        mask_roi = keep_largest_region(mask_roi)  # Keep only largest region

    else:
        return None

    if VISUAL_DEBUG:
        cv2.imshow("ROI", roi)
        cv2.imshow("MASK", mask_roi if mask_roi is not None else np.zeros_like(roi[:, :, 0]))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return mask_roi


# === MAIN LOOP ===
for img_path in image_files:
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Failed to load {img_path}")
        continue

    h, w = img.shape[:2]
    base_filename = os.path.splitext(os.path.basename(img_path))[0]
    results = model(img, verbose=False)[0]

    for i, det in enumerate(results.boxes):
        conf = det.conf.item()
        if conf < SCORE_THRESH:
            continue

        class_id = int(det.cls.item())
        class_name = model.names[class_id]
        xyxy = det.xyxy.cpu().numpy().squeeze().astype(int)
        xmin, ymin, xmax, ymax = np.clip(xyxy, 0, [w, h, w, h])
        bbox = (xmin, ymin, xmax, ymax)

        mask_roi = create_mask(img, bbox, class_name)

        mask_full = np.zeros((h, w), dtype=np.uint8)
        if mask_roi is not None:
            mask_full[ymin:ymax, xmin:xmax] = mask_roi
        else:
            print(f"[Fallback] No mask found, using bbox for {class_name}")
            cv2.rectangle(mask_full, (xmin, ymin), (xmax, ymax), 255, -1)

        class_dir = os.path.join(OUTPUT_ROOT, class_name)
        os.makedirs(class_dir, exist_ok=True)

        mask_filename = f"{base_filename}.png"
        cv2.imwrite(os.path.join(class_dir, mask_filename), mask_full)
        print(f"✅ {mask_filename} → {class_dir}/")

print("\n🎉 All masks created and sorted by class.")