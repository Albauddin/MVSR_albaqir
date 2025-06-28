# YOLO11s Training Pipeline

This directory contains code and instructions to **train a YOLOv8 object detection model** for the MVSR project.

## **Step-by-Step Instructions**

1. **Follow the official YOLOv8 training tutorial:**  
   [https://www.ejtech.io/learn/train-yolo-models](https://www.ejtech.io/learn/train-yolo-models)

2. **Prepare your dataset:**
   - Place all files from the provided `custom_data` (from Moodle) into the `custom_data` directory.

3. **Create a dataset YAML file:**
   - This script generates a `data.yaml` for training.
   ```bash
   python3 create_data_yaml.py

4. **Split data for training and validation:**
   - This script organizes the data for training/validation splits.
   ```bash
   python3 train_val_split.py

5. **Train YOLO11s model:**
   ```bash
   yolo detect train data=/workspace/data/YOLO/data.yaml model=yolo11s.pt epochs=85 imgsz=1048 name=train_11s degrees=10 scale=0.15 translate=0.05  

  - This script organizes the data for training/validation splits.

6. **Test trained model and count objects:**
   - Use this script for detection with a chosen confidence threshold.
   ```bash
   PYTHONPATH=./ultralytics python3 yolo_detect.py --model /workspace/data/YOLO/runs/detect/train_11s7/weights/best.pt --source test.jpg --thresh 0.8

7. **Generate segmentation masks for FoundationPose:**
   - This script produces object masks from your trained model outputs.
   ```bash
   python3 generate_masks.py

Note: Mask quality may not be perfect for grey objects, but is sufficient for use with FoundationPose.
Results are saved in the masks/ directory.




