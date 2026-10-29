# 🚗 Real-time Drivable Space Segmentation

Pixel-wise segmentation of **drivable road area** from front-camera frames, built for Level-4 autonomous-driving scenarios. A lightweight **UNet trained entirely from scratch** (no pretrained weights) on nuScenes urban street scenes.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![Hackathon](https://img.shields.io/badge/Hackathon-Manipal-0ea5e9?style=flat-square)

## Problem

A self-driving car must know where it can physically move — even when there are no lane markings. The task is binary segmentation (drivable vs. everything else: curbs, sidewalks, barriers) that is accurate at tricky boundaries such as road-to-grass edges and puddles, and fast enough for real-time use.

## Approach

| Stage | What happens |
|---|---|
| **Data** | nuScenes `CAM_FRONT` frames, resized to 256×256 |
| **Pseudo ground truth** | HSV colour filtering extracts grey road tones; top half of the frame is masked out to remove sky and buildings; morphological close/open cleans the mask |
| **Model** | Lightweight UNet encoder–decoder (16 → 32 filters) with skip connections, 1-channel output |
| **Training** | BCEWithLogitsLoss, Adam (lr 0.0005), 20 epochs, batch size 8 |
| **Inference** | Threshold at 0.3 → morphological cleanup (7×7) → crop sky region → green overlay blended on the original frame |

## Repository contents

| File | Description |
|---|---|
| `drivable-project (1).zip` | Source code: `dataset.py`, `model.py`, `train.py`, `test.py`, `main.py`, trained weights `model.pth`, and sample input/mask/overlay outputs |
| `CodeBusters_DrivableSpace (1).pptx` | Hackathon presentation: problem, pipeline, architecture and results |

## Run it

```bash
unzip "drivable-project (1).zip" && cd drivable-project
pip install torch opencv-python numpy
# point the dataset folder path in train.py / test.py to your nuScenes CAM_FRONT images
python train.py   # trains the UNet and saves model.pth
python test.py    # writes masks and overlays to outputs/
```

## Team Code Busters

Dakshraj Singh Chandawat (Team Lead & ML Engineer) · Aneeq T. · Keshav Heda · Inzamam Akmal

---
**Dakshraj Singh Chandawat** · [GitHub](https://github.com/DakshrajKurki) · [LinkedIn](https://www.linkedin.com/in/dakshraj-singh-chandawat-0a6349314)
