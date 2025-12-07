# Aadhaar Document Classification Pipeline

## Overview
This project builds an end-to-end **Aadhaar document classification system** using a two-stage hybrid ML pipeline:

1. **Stage 1 – Image Encoder (Vision Transformer)**  
   A pretrained Vision Transformer (from `timm`) extracts deep visual embeddings from Aadhaar card images.

2. **Stage 2 – OCR + Feature-based Classifier (Tesaract + XGBoost)**  
   OCR text is extracted using **Tesaract**, converted into handcrafted text features, and combined with image embeddings. An **XGBoost** classifier is trained on the concatenated features to produce high-accuracy predictions.

The notebook (`AIMLProject.ipynb`) contains the full pipeline: data loading, preprocessing, feature extraction, model training, evaluation, and single-image inference.

---

## Project Goals
- Robustly classify Aadhaar-related document images (e.g., valid Aadhaar card vs. invalid/other documents).
- Leverage both visual cues and textual content on the card for better accuracy.
- Provide a reproducible notebook-based implementation suitable for research and prototyping.

---

## Tech Stack
- Python 3.8+
- PyTorch
- timm (Vision Transformer models)
- EasyOCR, Tesract
- XGBoost
- NumPy, Pandas
- scikit-learn, matplotlib, seaborn

---

## Repository Structure
```
.
├─ AIMLProject.ipynb        # Primary Jupyter notebook with full pipeline
├─ README.md                # This README
├─ dataset/
│   ├─ train/
│   └─ test/
└─ models/                  # (optional) saved model artifacts
```

---

## Installation

Create a virtual environment (recommended) and install dependencies:

```bash
python -m venv venv
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate       # Windows

pip install --upgrade pip
pip install timm==0.9.2 torch torchvision pandas scikit-learn matplotlib xgboost easyocr seaborn
```

> Note: `torch` installation should match your CUDA / CPU setup. If you use CUDA, prefer the recommended command from https://pytorch.org.

---

## Notebook Walkthrough (AIMLProject.ipynb)
The notebook follows these logical steps:

1. **Setup & Imports**  
   Installs libs (if necessary), imports packages, sets file paths.

2. **Dataset & Transforms**  
   Defines `AadhaarDataset` (PyTorch `Dataset`) and image transforms:
   - Resize, center/resize crop
   - Normalize with ImageNet statistics (for pretrained ViT)

3. **Embedding Extraction (ViT)**  
   - Load a `timm` ViT model (pretrained)
   - Remove classification head and extract feature vectors per image
   - Save or cache embeddings to avoid repeated computation

4. **OCR Extraction (EasyOCR)**  
   - Run Tesaract on images (supports adding languages like 'hi' for Hindi)
   - Extract raw OCR text (joined lines)

## 5. Script Detection (OCR-Based Feature Engineering)

After extracting OCR text, the pipeline performs **script detection** to identify which language or script is present in the Aadhaar image.  
These script indicators are converted into numerical features and used by the classifier alongside visual embeddings.

   ### ✔ English Script Detection
   Detected when OCR output contains:
   - ASCII English letters (`A–Z`, `a–z`)
   - Common Aadhaar-related English keywords:
   - `UIDAI`
   - `INDIA`
   - `GOVT`
   - `AADHAAR`

   If matched → `is_english_script = 1`.

   ### ✔ Hindi (Devanagari) Script Detection
   Detected when OCR output contains characters in:
   - **Devanagari Unicode range:** `\u0900–\u097F`
   - Hindi Aadhaar-related keywords:
   - `भारत`
   - `आधार`

   If matched → `is_hindi_script = 1`.

   ### ✔ Other Regional Script Detection
   If OCR output contains characters in other Indic Unicode blocks:
   - Bengali: `\u0980–\u09FF`
   - Tamil: `\u0B80–\u0BFF`
   - (Optional extensions: Telugu, Kannada, Malayalam)

   If matched → `is_other_script = 1`.

   ### ✔ Additional Numeric Features from OCR Text
   These help assess text structure and formatting:
   - **`digit_ratio`** — proportion of numeric characters  
   - **`alpha_ratio`** — proportion of alphabetic characters  

   ### Final Script-Based Feature Set
   The final script detection features used for training are:

   - `is_english_script` (0 or 1)  
   - `is_hindi_script` (0 or 1)  
   - `is_other_script` (0 or 1)  
   - `digit_ratio` (float)  
   - `alpha_ratio` (float)

   These features are concatenated with **ViT image embeddings** and passed into the XGBoost classifier for final prediction.

6. **Feature Concatenation**  
   Concatenate image embeddings + OCR-derived features to create final training vectors.

7. **Classifier Training (XGBoost)**  
   - Train `xgboost.XGBClassifier` with hyperparameters:
     ```
     max_depth=6, n_estimators=300, learning_rate=0.05,
     subsample=0.7, colsample_bytree=0.7, objective="binary:logistic"
     ```
   - Evaluate using cross-validation or a holdout test set

8. **Evaluation & Metrics**  
   - Accuracy, precision, recall, F1-score
   - Confusion matrix visualization (matplotlib)
   - Example code to plot and save the confusion matrix included

9. **Inference / Predict Pipeline**  
   A `predict_pipeline(image_path)` function that:
   - Loads image, extracts ViT embedding
   - Runs Tesaract and computes text features
   - Concatenates features and returns predicted label + probability

---

## Example: Run Inference
```python
from your_notebook_or_module import predict_pipeline

sample = "/path/to/sample_adhar.png"
label, prob = predict_pipeline(sample)
print(f"Prediction: {label}  (prob: {prob:.4f})")
```

---

## Evaluation Example (confusion matrix)
```python
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

cm = confusion_matrix(true_labels, pred_labels)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.show()

print(classification_report(true_labels, pred_labels))
```

---

## Tips & Notes
- OCR quality highly impacts performance — consider language models or larger OCR configs for noisy images.
- Caching embeddings and OCR outputs drastically speeds up experimentation.
- For multilingual Aadhaar instances, add tesract languages (e.g., `'hi'` for Hindi).
- When moving to production, consider:
  - Converting the pipeline to a REST API (FastAPI)
  - Saving the feature extractor and model (joblib / pickle / ONNX)
  - Adding input validation and image size checks

---

## Future Improvements
- Add transformer-based text embeddings (BERT/multilingual-BERT) instead of handcrafted text features.
- Use model ensembling or a small trainable multimodal head for end-to-end fine-tuning.
- Add data augmentation and synthetic Aadhaar generation for better generalization.
- Deploy as a scalable microservice with GPU-backed inference.

---

## Contact
**Author:** Manshu Saini 
