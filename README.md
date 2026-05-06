# Labelling_Audio_Depth

## Overview

This project explores the prediction of perceived musical “depth” using machine learning, handcrafted audio features, and pretrained audio representations.

The work was based on and extended prior research on musical depth perception from the PSIC3839 dataset and related computational music cognition studies. The primary objective was to reproduce the original music depth recognition pipeline and build a working deployment interface capable of generating depth predictions for arbitrary uploaded songs.

The final system performs:

```text
Audio Upload
→ Audio Feature Extraction (librosa)
→ PCA Feature Reduction
→ Feature Reconstruction
→ Trained Regression Model
→ Predicted Musical Depth
```

A working Gradio interface was successfully developed for local deployment and real-time inference.

---

# Demo Video

A walkthrough of the project and live Gradio interface demonstration can be viewed here:

[Project Loom Demo](https://www.loom.com/share/513218866bd0495db8036f7f287c06ab)

---

# Presentation

Project presentation slides can be found here:

[Project Presentation](Labelling_Audio_Depth/reports/Labelling Audio Depth.pptx)

---

# Research Context

This project was built from the following foundational studies:

* *PSIC3839: Predicting the Overall Emotion and Depth of Entire Songs*
* *What a Deep Song: The Role of Music Features in Perceived Depth*
* *Associations Between Lyric and Musical Depth in Chinese Songs*

The original studies explored the relationship between perceptual musical depth and audio features such as:

* MFCCs
* Spectral contrast
* Spectral centroid
* Spectral bandwidth
* Spectral flatness
* Spectral rolloff
* Chromagrams
* Tonnetz features
* Tempo

The PSIC3839 dataset contains thousands of songs annotated for:

* arousal
* valence
* depth

using human perceptual ratings.

---

# Objectives

The goals of this project were to:

* Reproduce the original music depth recognition pipeline
* Explore perceptual music representation learning
* Build a weak supervision workflow using pseudo-label generation
* Compare handcrafted audio features against pretrained embeddings
* Deploy a working inference interface for real-time prediction

---

# Methodology

## 1. Teacher Model Training

Supervised regression models were trained on the PSIC3839 dataset using human-rated perceptual depth labels.

Target variables originally included:

* depth
* arousal
* valence

Final experiments focused primarily on musical depth prediction.

---

## 2. Weak Supervision / Teacher–Student Pipeline

A trained depth prediction model was deployed onto unlabeled Jamendo/MTG audio data in order to generate model-based labels for additional songs.

This workflow more closely resembles:

* weak supervision
* teacher–student learning
* knowledge transfer

rather than traditional pseudo-labeling.

---

## 3. Handcrafted Audio Feature Pipeline

Using `librosa`, the following features were extracted:

* MFCCs
* Spectral contrast
* Spectral centroid
* Spectral bandwidth
* Spectral flatness
* Spectral rolloff
* Chromagram
* Tonnetz
* Tempo

### Feature Engineering Pipeline

For each song:

1. Audio loaded and normalized
2. Spectrogram representations generated
3. Frame-level features extracted
4. High-dimensional features flattened
5. PCA applied separately to each feature block
6. Top 50 principal components retained per feature type
7. Features concatenated into final feature vector

Final vector:

```text
400 PCA-reduced features
+ 1 tempo feature
= 401 total features
```

---

# Models Evaluated

The following regression models were tested:

* Random Forest Regressor
* Gradient Boosting Regressor
* XGBoost Regressor

Evaluation metrics included:

* RMSE
* MAE
* R²
* Pearson Correlation Coefficient (PCC)

XGBoost produced the strongest performance among the handcrafted feature models.

---

# YAMNet Representation Learning Experiment

An additional experiment evaluated pretrained audio embeddings using Google’s YAMNet model.

Workflow:

```text
Audio
→ YAMNet Embeddings
→ XGBoost Regressor
→ Depth Prediction
```

Results demonstrated that pretrained audio may improved generalization performance compared to handcrafted spectral features. However, due to lack of ground truth labels, this cannot be certain.

Approximate results using weak labels:

```text
PCC ≈ 0.77
RMSE ≈ 0.108
```

This suggests pretrained audio representations may capture perceptual musical depth more effectively under weak supervision conditions.

---

# Key Engineering Challenges

## Data Leakage

A train/test contamination issue was discovered where portions of testing data appeared within training samples.

This caused inflated evaluation metrics.

The leakage issue was corrected and evaluation metrics normalized afterward.

---

## Deployment Pipeline Separation

One major lesson from this project was distinguishing between:

* training pipelines
* inference pipelines

Deployment artifacts were saved independently, including:

* trained regression model
* PCA transformers
* feature column ordering

---

## PCA Feature Mismatch Bug

A major deployment bug occurred due to PCA artifacts being trained on CSVs containing accidental index columns.

This produced a feature mismatch:

```text
Expected: 777 features
Received: 776 features
```

The issue was resolved by correctly loading CSVs using:

```python
pd.read_csv(..., index_col=0)
```

---

# Deployment

A working local deployment interface was developed using Gradio.

Capabilities:

* Drag-and-drop audio upload
* Real-time prediction
* librosa preprocessing
* PCA transformation
* XGBoost inference pipeline

Current deployment workflow:

```text
Audio Upload
→ Feature Extraction
→ PCA Transform
→ 401-Feature Reconstruction
→ Depth Prediction
```

Future deployment goals include:

* FastAPI backend
* Docker containerization
* HuggingFace Spaces deployment
* Render deployment
* Cloud inference APIs

---

# Repository Structure

```text
Labelling_Audio_Depth/
│
├── notebooks/
│   ├── 01_PSIC_model.ipynb
│   ├── 02_MTG_audio_extractions.ipynb
│   ├── 03_MTG_feature_pipeline.ipynb
│   ├── 04_deploy_pseudolabel.ipynb
│   ├── 05_testing_pseudolabelling.ipynb
│   ├── 06_comparing_MTG_YAMNet.ipynb
│   └── 07_depth_model_interface.ipynb
│
├── models/
│   ├── depth_model.joblib
│   ├── PCA transformers
│   └── feature_columns.joblib
│
├── references/
│   ├── research papers
│   └── literature
│
├── reports/
│   ├── presentations
│   └── summaries
│
├── requirements.txt
└── README.md
```

---

# Technologies Used

* Python
* librosa
* scikit-learn
* XGBoost
* TensorFlow
* TensorFlow Hub
* YAMNet
* pandas
* NumPy
* Gradio
* Jupyter Notebook

---

# Future Work

Potential future improvements include:

* FastAPI deployment
* Dockerized inference service
* GPU inference optimization
* Batch prediction pipelines
* Retrieval-based audio similarity search
* Transformer-based audio representation models
* Comparison against additional pretrained audio encoders
* Integration of lyric-based NLP features with audio embeddings

---

# References

Xu, L. et al. (2022). *PSIC3839: Predicting the Overall Emotion and Depth of Entire Songs.*

Wen, X. et al. (2022). *What a Deep Song: The Role of Music Features in Perceived Depth.*

Xu, L. et al. (2024). *Associations Between Lyric and Musical Depth in Chinese Songs.*
