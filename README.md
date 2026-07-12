# 🌿 Crop Disease Detection using MobileNetV2

> **Transfer Learning · Deep Learning · Computer Vision · Model Explainability**

A deep learning system that detects and classifies **tomato plant diseases** from leaf images using **MobileNetV2 transfer learning**. Trained on the PlantVillage dataset, the model classifies leaf images into **10 disease categories** with GradCAM-based explainability to highlight exactly which part of the leaf influenced the prediction.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Disease Classes](#disease-classes)
- [Project Pipeline](#project-pipeline)
- [Model Architecture](#model-architecture)
- [Training Strategy](#training-strategy)
- [Results](#results)
- [GradCAM Explainability](#gradcam-explainability)
- [How to Run](#how-to-run)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)

---

## Overview

Farmers lose significant crop yields every year due to late or inaccurate manual diagnosis of plant diseases. This project addresses that problem with an AI-powered image classification system that can identify tomato leaf diseases from a single photo — instantly and accurately.

The model uses **MobileNetV2** (pre-trained on ImageNet) as a feature extractor, then adds a custom classification head and fine-tunes the top layers for the specific task of tomato disease classification.

**Key highlights:**
- 18,000+ leaf images across 10 disease classes
- Two-phase training: feature extraction → fine-tuning
- 7 data augmentation techniques to improve generalisation
- GradCAM visualisations for model interpretability
- Top-3 confidence score output per prediction

---

## Disease Classes

The model classifies tomato leaves into the following 10 categories:

| # | Class | Type |
|---|-------|------|
| 1 | Bacterial Spot | Disease |
| 2 | Early Blight | Disease |
| 3 | Late Blight | Disease |
| 4 | Leaf Mold | Disease |
| 5 | Septoria Leaf Spot | Disease |
| 6 | Spider Mites (Two-spotted) | Disease |
| 7 | Target Spot | Disease |
| 8 | Tomato Yellow Leaf Curl Virus | Disease |
| 9 | Tomato Mosaic Virus | Disease |
| 10 | Healthy | Healthy |

---

**Why MobileNetV2?**
- Lightweight and computationally efficient
- Excellent accuracy-to-parameter ratio
- Well-suited for transfer learning on image classification tasks
- Pre-trained on ImageNet provides powerful low-level feature extraction

---

## Training Strategy

### Phase 1 — Feature Extraction
| Parameter | Value |
|-----------|-------|
| Base Model | Frozen |
| Epochs | 20 |
| Learning Rate | 1e-3 (Adam) |
| Batch Size | 32 |
| Loss | Categorical Cross-Entropy |

### Phase 2 — Fine-Tuning
| Parameter | Value |
|-----------|-------|
| Layers Unfrozen | MobileNetV2 layers after layer 100 |
| Epochs | 10 (continuing from Phase 1) |
| Learning Rate | 1e-5 (Adam) |

### Callbacks
| Callback | Monitor | Purpose |
|----------|---------|---------|
| `EarlyStopping` | val_accuracy (patience=5) | Stop when improvement stalls |
| `ModelCheckpoint` | val_accuracy | Save best weights automatically |
| `ReduceLROnPlateau` | val_loss (patience=3, factor=0.3) | Reduce LR on plateau |

### Data Augmentation (Training Only)
| Technique | Value |
|-----------|-------|
| Rotation | ±40° |
| Width Shift | 20% |
| Height Shift | 20% |
| Shear | 20% |
| Zoom | 20% |
| Brightness | 0.8 – 1.2 |
| Horizontal Flip | Enabled |

---

## Results

| Metric | Value |
|--------|-------|
| **Dataset** | PlantVillage (Kaggle) |
| **Total Images** | 18,000+ |
| **Classes** | 10 tomato disease categories |
| **Train / Val / Test Split** | 80% / 10% / 10% |
| **Input Size** | 224 × 224 × 3 |
| **Evaluation** | Per-class F1-Score, Confusion Matrix |
| **Output** | Top-3 predictions with confidence scores |

> **Note:** Run `model.evaluate(test_generator)` after training to get your exact test accuracy and loss values.


## GradCAM Explainability

**Gradient-weighted Class Activation Mapping (GradCAM)** highlights the regions of the leaf image the model focused on when making its prediction — making the model transparent and trustworthy rather than a black box.

The implementation drills into the MobileNetV2 sub-model to access the `Conv_1` layer (the last convolutional layer) and computes gradients with respect to the predicted class.

```python
# GradCAM accesses the base model's Conv_1 layer
base_model = model.get_layer('mobilenetv2_1.00_224')
last_conv_layer = base_model.get_layer('Conv_1')
```

**Output for each image:**
- **Original image** — raw input
- **GradCAM heatmap** — regions of interest (hot = high activation)
- **Overlay** — heatmap superimposed on original with prediction and confidence

---

## How to Run

### Prerequisites
- Google Colab (recommended, free GPU)
- Kaggle account with API token (`kaggle.json`)

### Step 1 — Open in Google Colab
Upload `Crop_Disease_Detection_MobileNetV2.ipynb` to [Google Colab](https://colab.research.google.com/) and enable GPU:

```
Runtime → Change runtime type → T4 GPU
```

### Step 2 — Upload Kaggle Token
When Cell 2 runs, it will prompt you to upload a file. Upload your `kaggle.json`:

1. Go to [kaggle.com](https://kaggle.com) → Settings → API → **Create New Token**
2. Upload the downloaded `kaggle.json` when prompted in Colab

### Step 3 — Run All Cells
```
Runtime → Run all
```

The notebook will automatically:
- Download and extract the PlantVillage dataset
- Filter tomato classes
- Run EDA visualisations
- Split and augment the data
- Train Phase 1 (frozen base)
- Fine-tune Phase 2
- Evaluate on the test set
- Generate GradCAM outputs
- Save the model as `tomato_disease_detector.keras`

### Step 4 — Predict on a New Image
```python
predict_disease("path/to/your/leaf_image.jpg", model, class_names)
```
This outputs the top-3 predicted disease classes with confidence scores as a visual bar chart.

---

## Project Structure

```
Crop-Disease-Detection/
│
├── Crop_Disease_Detection_MobileNetV2.ipynb   # Main notebook
├── class_names.json                           # Class label mapping (generated)
├── tomato_disease_detector.keras              # Saved model (generated)
├── README.md                                  # This file
│
├── outputs/                                   # Generated visualisations
│   ├── class_distribution.png
│   ├── sample_images.png
│   ├── training_curves_Phase_1.png
│   ├── training_curves_Phase_2.png
│   ├── confusion_matrix.png
│   ├── gradcam_output.png
│   └── prediction_output.png
```

---

## Tech Stack

| Category | Tools |
|----------|-------|
| **Language** | Python 3.10+ |
| **Deep Learning** | TensorFlow 2.x, Keras |
| **Model** | MobileNetV2 (ImageNet weights) |
| **Data Processing** | Pandas, NumPy, split-folders |
| **Visualisation** | Matplotlib, Seaborn |
| **Evaluation** | Scikit-learn |
| **Environment** | Google Colab (GPU) |
| **Dataset** | PlantVillage (Kaggle) |

---

## Author

**Kavindu Madusara Disanayaka**
