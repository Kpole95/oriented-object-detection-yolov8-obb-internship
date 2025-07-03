# Oriented Object Detection for Houses and Tennis Courts using YOLOv8-OBB

This repository includes an implementation of an Oriented Object Detection (OBB) model. It is specifically designed to detect houses and tennis courts in aerial images and estimate their precise rotation angles. This project was developed during an AI/ML internship assessment.

## Objective

The main goal was to implement and evaluate an object detection model that can handle object orientation. It predicts both bounding boxes and rotation angles for "Tennis Courts" and "Houses."

## Project Highlights

* **Custom Dataset Annotation:** Carefully annotated 90 aerial images by hand with individual, rotated bounding boxes for target objects using Roboflow.
* **YOLOv8-OBB Architecture:** Fine-tuned a pre-trained `yolov8n-obb.pt` model, which is a leading solution for oriented object detection.
* **Thorough Evaluation:** Evaluated model performance using standard mAP metrics for bounding box accuracy and a custom Python script for Mean Absolute Angle Error for orientation accuracy.
* **Solid Implementation:** Successfully addressed complex setup, dependency management, and data formatting challenges such as 8-point OBB label parsing.

## Key Results

The fine-tuned YOLOv8-OBB model showed promising results:

| Metric                     | All Objects | House   | Tennis Court |
| :------------------------- | :---------- | :------ | :----------- |
| **mAP50**                 | 0.858       | 0.844   | 0.872        |
| **mAP50-95**              | 0.601       | 0.478   | 0.725        |

**Mean Absolute Angle Error (Test Set): 51.91 degrees** (measured on 67 matched instances).
*(Note: An angle error of 51.91 degrees on a 0-180 degree scale represents about 29% of the full angular range.)*

## Visual Examples

Here are a few examples of model predictions from the test set:

*(**Replace these placeholders with actual images.** You can drag and drop on GitHub's README editor or use image paths like `runs/obb/predictX/P0463_png.rf.eda446f96e380a0a67a51e094a22ac36.jpg`)*

### Successful Detections:

![Successful Detection - Tennis Courts](https://github.com/Kpole95/oriented-object-detection-yolov8-obb-internship/blob/main/runs/obb/predict2/P0463_png.rf.eda446f96e380a0a67a51e094a22ac36.jpg)

*Caption: Model prediction showing multiple tennis courts with accurate rotated bounding boxes and high confidence scores.*

![Successful Detection - Houses](https://github.com/Kpole95/oriented-object-detection-yolov8-obb-internship/blob/main/runs/obb/predict2/P0444_png.rf.3cab4de975fdd5d1f8d91d4e8b5d6da7.jpg)

*Caption: Successful detection of several houses in a varied environment, accurately localized with properly oriented bounding boxes.*

### Challenging Cases:

![Challenging Case - Missed Detections](https://github.com/Kpole95/oriented-object-detection-yolov8-obb-internship/blob/main/runs/obb/predict2/P0451_png.rf.e64067be4d7a25530e26df81d3bf14e7.jpg)

*Caption: A challenging scenario where only one of several tennis courts is detected, and its orientation is simplified to horizontal, showing missed detections and a need for improvement.*

---

## Full Report

For a detailed overview of the project methodology, evaluation results, and further insights, please refer to the full report:

[https://github.com/Kpole95/oriented-object-detection-yolov8-obb-internship/blob/main/reports/Oriented%20Object%20Detection%20for%20Houses%20and%20Tennis%20Courts%20using%20YOLOv8-Murali%20Krishna%20Pole.pdf]

## Acknowledgments

This project was completed as part of an AI/ML internship assessment.

---
