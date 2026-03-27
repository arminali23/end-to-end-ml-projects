# ML Projects

A collection of end-to-end machine learning projects I'm building to go from ML fundamentals to production-grade systems — covering classical ML, deep learning, computer vision, FastAPI, and Docker.

Each project is fully containerized and served as a REST API. No notebooks. Real structure, real deployment patterns.

---

## Projects

### 01 · Titanic Survival Predictor
A binary classification API that predicts passenger survival. Built with a full scikit-learn pipeline (preprocessing + model in one object) served via FastAPI and containerized with Docker.

**Key concepts:** `scikit-learn Pipeline` · `ColumnTransformer` · `RandomForestClassifier` · `Pydantic validation` · `FastAPI` · `Docker`

---

### 02 · House Price Predictor
A regression API that predicts median house prices. Automatically compares multiple models using cross-validation and selects the best one before saving. Ships an evaluation report alongside the model artifact.

**Key concepts:** `Regression` · `Feature engineering` · `Model comparison` · `Cross-validation` · `RMSE / R²` · `GradientBoosting` · `FastAPI` · `Docker`

---

### 03 · Image Classifier API
A computer vision API that accepts an image upload and returns top-5 predictions using a pretrained ResNet50 model. Covers the full deep learning inference pipeline including image preprocessing and ImageNet normalization.

**Key concepts:** `PyTorch` · `Transfer learning` · `ResNet50` · `ImageNet` · `Image preprocessing` · `File uploads in FastAPI` · `Docker`

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-containerized-2496ED?style=flat&logo=docker&logoColor=white)

---

## Structure

```
ml-projects/
├── titanic-ml-api/
├── house-price-api/
└── image-classifier-api/
```

Each project is self-contained with its own `Dockerfile`, `requirements.txt`, and `app/` module.
