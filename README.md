# MURA Radiograph Classifier: Bone Fracture Detection
[![Hugging Face Model](https://img.shields.io/badge/🤗%20Hugging%20Face-Download%20Model-blue)](https://huggingface.co/subhan1501/MURA-EfficientNetV2-Fracture-Detection)

### Deep Learning | Medical Image Analysis | EfficientNetV2

This project focuses on automating the detection of bone abnormalities. Utilizing the **MURA (Musculoskeletal Radiographs)** dataset, the pipeline implements a state-of-the-art **EfficientNetV2** architecture via transfer learning to classify X-rays into binary categories: `fractured` or `not_fractured`. 

---

# Key Implementation Details

## Data Engineering & Pipeline
Working with raw medical data requires rigorous preprocessing. This repository includes custom logic to handle the MURA dataset:
* **Data Cleansing**: Implemented a PIL-based verification script to detect and automatically remove corrupted image files (`img.verify()`) prior to generator ingestion.
* **Stratified Splitting**: Automatically orchestrates the reorganization of unstructured folders into clean `train`, `val`, and `test` directories using a stratified `train_test_split` to maintain class balance.
* **Data Augmentation**: Deployed `ImageDataGenerator` to perform real-time spatial transformations (10° rotation, shifting, horizontal flipping, zooming) and applied specific `efficientnet_v2.preprocess_input` scaling.

## Network Architecture
* **Base Model**: Leveraged **EfficientNetV2**, recognized for its optimized parameter efficiency and high accuracy in fine-grained image classification tasks.
* **Classification Head**: Appended a `GlobalAveragePooling2D` layer feeding into a custom `Dense` network tailored for binary probability scoring.
* **Optimization**: The model was configured for a 224x224 input resolution and compiled using a custom learning rate strategy (`1e-4`) over 25 epochs.

## Evaluation & Diagnostics
The system provides a comprehensive suite of clinical evaluation metrics:
* **Confusion Matrix & Classification Report**: Analyzes Precision, Recall, and F1-Scores across both target classes.
* **ROC-AUC Analysis**: Engineered custom Matplotlib visualizations to plot the Receiver Operating Characteristic (ROC) curve and calculate the Area Under the Curve (AUC) for threshold evaluation.

---

# ow to Use (Inference)
You can download the pre-trained weights from Hugging Face and run inference on any X-ray image using the provided `predict.py` script:

```bash
python predict.py --image path/to/xray.png --model MURA_EfficientNetV2L.h5