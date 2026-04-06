# 🚀 Project Context: Offroad Semantic Scene Segmentation (Duality AI Hackathon)

---

## 1. Problem Overview

This project is part of the **Duality AI Offroad Autonomy Segmentation Challenge**.

The objective is to:
- Train a **semantic segmentation model** on synthetic desert data
- Test it on **unseen desert environments**
- Optimize for **accuracy (IoU)** and **generalization**

All data is generated using a **digital twin simulation platform (Falcon)**.

---

## 2. Core Objective

Build a model that:
- Takes an RGB image as input
- Outputs a **pixel-wise segmentation map**

---

## 3. Classes to Predict

| ID    | Class Name        |
|-------|------------------|
| 100   | Trees            |
| 200   | Lush Bushes      |
| 300   | Dry Grass        |
| 500   | Dry Bushes       |
| 550   | Ground Clutter   |
| 600   | Flowers          |
| 700   | Logs             |
| 800   | Rocks            |
| 7100  | Landscape        |
| 10000 | Sky              |

---

## 4. Dataset Structure

dataset/
│
├── train/
│   ├── images/
│   ├── masks/
│
├── val/
│   ├── images/
│   ├── masks/
│
├── testImages/
│   ├── *.png (NO labels)

⚠️ Do NOT use testImages during training

---

## 5. Workflow

1. Load dataset  
2. Preprocess images  
3. Train model  
4. Validate performance  
5. Test on unseen images  
6. Compute IoU  
7. Optimize  

---

## 6. Key Metric

IoU (Intersection over Union) → maximize this

---

## 7. Deliverables

- Trained model
- train.py & test.py
- Config files
- Report (IoU, loss, failure cases)
- README.md

---

## 8. Constraints

- Use only provided dataset  
- No external data  
- No test data leakage  

---

## 9. Tech Stack

- Python  
- PyTorch  
- OpenCV  
- NumPy  

Models:
- UNet  
- DeepLabV3+  
- SegFormer  
- YOLOv8-seg  

---

## 10. Optimization Goals

- Improve IoU  
- Handle class imbalance  
- Improve generalization  
- Reduce inference time  

---

## 11. Common Challenges

- Low IoU → tuning needed  
- Overfitting → regularization  
- Class imbalance → weighted loss  
- Slow training → optimize model  

---

## 12. Agent Instructions

- Treat as semantic segmentation task  
- Focus on IoU  
- Avoid test data leakage  
- Write clean, modular, reproducible code  

---

## 13. Success Criteria

- High IoU  
- Good generalization  
- Efficient model  
- Clean implementation  

---

## 🧠 End Goal

Build a robust segmentation model for unseen environments.
