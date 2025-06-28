# FoundationPose

This folder contains the setup and scripts for **6D object pose estimation using FoundationPose**.

## Tutorial Reference

Before using this code, **please follow the official FoundationPose setup and tutorial** at:  
[https://github.com/NVlabs/FoundationPose/tree/main](https://github.com/NVlabs/FoundationPose/tree/main)

This will help you set up all dependencies, datasets, and model files required to run the pipeline.

---

## Key Files

- **`run_multi_demo.py`**  
  The main entry point to run pose estimation on multiple scenes or objects.

---

## Example Usage

```bash
python run_multi_demo.py \
  --test_scene_dir /home/imamzen/Documents/Docker/mvsr/FoundationPose/demo_data/mustard0 \
  --debug_dir /home/imamzen/Documents/Docker/mvsr/FoundationPose/debug
