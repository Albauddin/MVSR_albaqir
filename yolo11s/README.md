# YOLO11s Training Pipeline

This directory contains code and instructions to **train a YOLOv8 object detection model** for the MVSR project.

## **Step-by-Step Instructions**

1. **Follow the official YOLO training tutorial:**  
   [https://www.ejtech.io/learn/train-yolo-models](https://www.ejtech.io/learn/train-yolo-models)

2. **Prepare your dataset:**
   - Place all files from the provided `custom_data` (from Moodle) into the `custom_data` directory.

3. **Create a dataset YAML file:**
   - This script generates a `data.yaml` for training.

   ```bash
   python3 create_data_yaml.py

4. **Split data for training and validation:**
   - This script organizes the data for training/validation splits. (provided from tutorial)

   ```bash
   python3 train_val_split.py

5. **Train YOLO11s model:**

   ```bash
   yolo detect train data=/workspace/data/YOLO/data.yaml model=yolo11s.pt epochs=85 imgsz=1048 name=train_11s degrees=10 scale=0.15 translate=0.05  

  - epochs = 85 with Augmented data as above with the degree, scale and translate variation shows the best result for the trained model

6. **Test trained model and count objects:**
   - Use this script for detection with a chosen confidence threshold. (provided from tutorial)

   ```bash
   PYTHONPATH=./ultralytics python3 yolo_detect.py --model /workspace/data/YOLO/runs/detect/train_11s7/weights/best.pt --source test.jpg --thresh 0.8

7. **Generate segmentation masks for FoundationPose:**

This script produces object masks from your trained model outputs.
method being used for this script is:
   - Loads a trained YOLO model to detect objects in all images from a directory.
   - For each detected object above a confidence threshold:
      - Extracts the bounding box region (ROI) for that object.
      - For “yellow” objects: creates a mask using HSV color thresholding.
      - For “gray/grey/grau/black” objects: creates a mask based on pixel similarity in BGR channels (with an Otsu fallback for weak cases).
      - Cleans the mask with morphological operations and keeps only the largest region.
   - Applies the object mask into the correct position within the full-size image.
   - Saves all masks as .png images, sorted into class-named subfolders in the output directory.
   - Prints progress for each processed image and reports when done.
Note: Mask quality may not be perfect for grey objects, but is sufficient for use with FoundationPose. Results are saved in the masks/ directory.

   ```bash
   python3 generate_masks.py
   ```

the resulted masks will be in the masks directory like this and can be renamed to be used in the foundationpose

   ```bash
masks/
├── Left_Frame_Grey/
├── Left_Frame_Yellow/
├── Left_Frame_Yellow_2/
├── Right_Frame_Yellow/
├── Right_Frame_Yellow_2/
└── Special_Left_Frame_Grey/
   ```
