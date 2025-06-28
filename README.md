# MVSR Project: 6D Object Pose Estimation Pipeline

This repository contains a full pipeline for 6D object pose estimation, originally organized as two separate containers:

- **YOLO (for object detection):**  
  Based on [this tutorial](https://www.ejtech.io/learn/train-yolo-models)

- **FoundationPose (for 6D pose estimation):**  
  See [the official FoundationPose repository](https://github.com/NVlabs/FoundationPose/tree/main)

---

## Recommended Usage

While the pipeline was initially structured to use both YOLO and FoundationPose containers, **it is now recommended to start directly with the FoundationPose tutorial**. This allows you to run the project using just a single container, streamlining setup and usage.

- See the `foundationpose/` folder for setup and instructions.
- The `yolo11s/` folder is provided for reference and reproducibility.

---
