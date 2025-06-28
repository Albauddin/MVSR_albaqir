import os
import json
import argparse
import cv2
import numpy as np

# === Dummy Pose Estimation ===
def dummy_pose_estimation():
    R = np.eye(3)
    t = np.array([0, 0, 0.5])  # 50cm in front of the camera
    return R, t

# === Visualize pose with projected axes ===
def draw_axes(image, K, R, t, length=50):
    axis_3d = np.float32([
        [0, 0, 0],
        [length, 0, 0],
        [0, length, 0],
        [0, 0, length]
    ])
    dist_coeffs = np.zeros((4, 1))
    axis_2d, _ = cv2.projectPoints(axis_3d, cv2.Rodrigues(R)[0], t, K, dist_coeffs)
    axis_2d = axis_2d.reshape(-1, 2).astype(int)
    origin = tuple(axis_2d[0])
    image = cv2.line(image, origin, tuple(axis_2d[1]), (0, 0, 255), 2)
    image = cv2.line(image, origin, tuple(axis_2d[2]), (0, 255, 0), 2)
    image = cv2.line(image, origin, tuple(axis_2d[3]), (255, 0, 0), 2)
    return image

# === Load class list ===
def load_classes(class_file):
    with open(class_file, 'r') as f:
        return [line.strip() for line in f.readlines() if line.strip()]

# === Extract class name from filename ===
def extract_class_name(filename):
    name = os.path.splitext(os.path.basename(filename))[0]
    parts = name.split("_", 2)
    return parts[2].replace("_", " ") if len(parts) == 3 else "Unknown"

# === Main ===
def main(args):
    with open(args.intrinsics, 'r') as f:
        intr = json.load(f)
    K = np.array([
        [intr['fx'], 0, intr['cx']],
        [0, intr['fy'], intr['cy']],
        [0, 0, 1]
    ])

    rgb = cv2.imread(args.rgb)
    assert rgb is not None, "RGB image could not be loaded"

    depth = cv2.imread(args.depth, cv2.IMREAD_UNCHANGED)
    assert depth is not None, "Depth image could not be loaded"

    class_list = load_classes("custom_data/classes.txt")
    print("Loaded classes:", class_list)

    class_name = extract_class_name(args.rgb)
    print(f"✅ Detected class: {class_name}")

    R, t = dummy_pose_estimation()
    output = draw_axes(rgb.copy(), K, R, t)

    output_path = os.path.join("results", f"pose_overlay_{class_name.replace(' ', '_')}.png")
    cv2.imwrite(output_path, output)
    print(f"✅ Pose overlay saved at: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb", required=True, help="Path to cropped RGB image")
    parser.add_argument("--depth", required=True, help="Path to cropped depth image")
    parser.add_argument("--intrinsics", required=True, help="Path to camera intrinsics JSON")
    args = parser.parse_args()
    main(args)

