# FoundationPose

This folder contains the setup and scripts for **6D object pose estimation using FoundationPose**.

## Tutorial Reference

Before using this code, **please follow the official FoundationPose setup and tutorial** at:  
[https://github.com/NVlabs/FoundationPose/tree/main](https://github.com/NVlabs/FoundationPose/tree/main)

This will help you set up all dependencies, datasets, and model files required to run the pipeline.

---

## Key Files

- **`run_multi_demo.py`**  
In run_multi_demo.py, multiple object pose estimations are performed per frame by looping through all object classes detected in the scene:
  1. For each frame:
    - The script loads the RGB image and depth map.
  2. For each object class (e.g., "Left_Frame_Grey", "Right_Frame_Yellow"):
    - It loads the corresponding segmentation mask and 3D mesh model for that object in the current frame.
    - If both mask and mesh exist, it runs the FoundationPose pose registration and refinement pipeline for this object.
    - The estimated pose is used to draw a 3D box and axes representing the object's position and orientation on the canvas.
  3. The script does this sequentially for all object classes in that frame, drawing each pose on the same image.
  4. After all objects in the frame are processed, the resulting image (with all estimated poses overlaid) is saved for that frame.

---

## Example Usage

```bash
python run_multi_demo.py \
  --test_scene_dir /path_to_directory/FoundationPose/demo_data/mustard0 \
  --debug_dir /path_to_directory/FoundationPose/debug
