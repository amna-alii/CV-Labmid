# CV-Labmid
# Blood Cell Classification: HOG + SVM vs SIFT + SVM

**Computer Vision Lab Mid Exam**
**Name:** Amna Ali | **Registration No.:** FA23-BAI-060

A small lab wants a simple blood-cell classifier that runs on a normal laptop. This project builds two classical pipelines, **HOG + LinearSVC** and **SIFT + LinearSVC**, and tests which one is more reliable when images are **rotated** or **noisy**.

## Repository contents

| File | Description |
|---|---|
| `FA23-BAI-060_AmnaAli_LabMid.ipynb` | Executed notebook (Google Colab) with all outputs visible (GitHub shows them when you open the file) |
| `README.md` | Project summary and results |

## Personal settings (from registration number 060)

| Setting | Rule | Value |
|---|---|---|
| Random seed | Last two digits | **60** |
| HOG cell size | Last digit even → 8×8 | **8 × 8** |
| SIFT grid size | Last digit 0–4 → 4×4 | **4 × 4** (16 keypoints) |

## Fixed settings (same for everyone)

- **Dataset:** BloodMNIST (`pip install medmnist`), image size 64
- **Classes:** eosinophil (1), monocyte (5), neutrophil (6), platelet (7)
- **Images per class:** 100 train, 30 validation, 30 test (chosen randomly with the seed)
- **Preparation:** grayscale, resized to 128 × 128
- **HOG:** 9 orientations, 2 × 2 cells per block
- **SIFT:** fixed grid of keypoints, keypoint size 32
- **Classifier:** `LinearSVC`, C chosen from {0.1, 1, 10} using the validation set only
- **Rotation test:** 20 degrees
- **Noise test:** Gaussian noise, sigma = 0.05 (images scaled to 0–1)

## Method

### HOG + SVM
1. Convert each image to grayscale 128 × 128.
2. Extract HOG features (8×8 cells, 2×2 cells per block, 9 orientations).
   Feature length = 15 × 15 blocks × 36 values = **8100**.
3. Train `LinearSVC` for each C, choose C on the validation set.

### SIFT + SVM
1. Normal SIFT (`detectAndCompute`) gives a different number of keypoints per image, which an SVM cannot use directly.
2. Fix: place a **4 × 4 grid** of keypoints (size 32) and compute descriptors with `sift.compute`.
   Feature length = 16 keypoints × 128 = **2048**.
3. Train `LinearSVC` for each C, choose C on the validation set.

### Robustness (Q4)
Both selected models are frozen. The transformation (rotation or noise) is applied to the **test images first**, then features are extracted, then the frozen model predicts.

### No data leakage
- Train data trains the model.
- Validation data selects C.
- Test data is used **only in Q4**.

## Results

All values below come from the executed notebook.

### Validation results (selected C = 0.1 for both models)

| Model | Feature length | Best C | Validation accuracy | Validation macro-F1 |
|---|---|---|---|---|
| HOG + SVM | 8100 | 0.1 | 0.7417 | 0.7385 |
| SIFT + SVM | 2048 | 0.1 | 0.7500 | 0.7540 |

### Test results (Q4 robustness)

| Model | Clean accuracy | Rotated (20°) accuracy | Noisy (σ = 0.05) accuracy |
|---|---|---|---|
| HOG + SVM | 0.7417 | 0.7000 | 0.5333 |
| SIFT + SVM | 0.7500 | 0.4917 | 0.7083 |

Accuracy drop from clean:

| Model | Drop under rotation | Drop under noise |
|---|---|---|
| HOG + SVM | 0.0417 | 0.2083 |
| SIFT + SVM | 0.2583 | 0.0417 |

### Most confused classes (validation set)

- HOG + SVM: **eosinophil and neutrophil** (16 mistakes)
- SIFT + SVM: **eosinophil and monocyte** (16 mistakes)

## Key findings

- On clean images the two models perform similarly.
- **HOG** was more reliable under **rotation**.
- **SIFT** was more reliable under **noise**.
- Neither model was best in every condition.

## Extra analysis added

- Per-class classification report (precision, recall, F1) for both models on the validation set.
- Display of wrongly classified validation images with true and predicted labels.

## Limitations

- The test set has only 120 images (30 per class), so one image is about 0.83 percentage points of accuracy.
- The noise result uses a single random noise draw (seed 60).
- Reasons for why each model reacts differently to rotation and noise were not tested in this project.

## How to run

1. Open the notebook in Google Colab.
2. Click **Runtime → Run all** (the notebook installs `medmnist` in its first cells).
3. Libraries used: `medmnist`, `numpy`, `opencv-python`, `scikit-image`, `scikit-learn`, `matplotlib`, `pandas`.

## AI usage

An AI tool was used to generate the starting code. The code was reviewed, run and checked by the student. See the AI log table at the end of the notebook for the prompt, the changes made, and the verification steps.
