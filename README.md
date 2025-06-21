# Human Detection and Attribute Extraction System

This project develops a deep learning-based system to **detect humans** in images or videos and **extract key attributes** such as **posture** (standing/sitting) and **hair color** (black, brown, blonde, other). The system combines **ResNet18** for classification and **YOLOv8** for detection, working effectively on a custom-labeled dataset.

## Objectives

- Detect humans in images or video streams
- Classify posture: `standing` or `sitting`
- Classify hair color: `black`, `brown`, `blonde`, or `other`

## System Architecture

### Data Preprocessing

- Label images manually with: `posture` and `hair_color`
- Resize images to `224x224`
- Apply augmentations: flipping, rotation, brightness adjustment
- Split into: `train.csv`, `val.csv`, `test.csv`

### Deep Learning Model

- Backbone: ResNet18 (pretrained)
- Two classification heads:
  - `fc_pose`: classify posture
  - `fc_hair`: classify hair color
- Loss Function: CrossEntropyLoss
- Optimizer: Adam

### Inference Pipeline

- Image Inference: classify posture and hair color of a single image
- Video Inference: detect humans with YOLOv8, then classify attributes with ResNet18

## Project Structure

