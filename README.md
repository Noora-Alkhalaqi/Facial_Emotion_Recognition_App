# Facial Emotion Recognition App

A Streamlit web app that predicts the emotion on a face from an uploaded image or a webcam snapshot. The model is a simple fully connected neural network (Keras) trained on the [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013) dataset.

## Features

- Upload a face image (`.jpg`, `.jpeg`, `.png`) or take a picture with your webcam
- Shows the 48x48 grayscale image the model actually sees
- Displays the predicted emotion, its confidence, and a bar chart of the probabilities for all 7 classes

**Emotion classes:** angry, disgust, fear, happy, neutral, sad, surprise

## Project Structure

```
.
├── app.py                 # Streamlit app
├── DL_lap1.ipynb          # Notebook: data loading, training, and evaluation (Google Colab)
├── emotion_model.keras    # Trained model used by the app
└── README.md
```

## Getting Started

### Requirements

- Python 3.10+
- `streamlit`, `keras`, `tensorflow`, `numpy`, `pandas`, `pillow`

### Installation

```bash
git clone <repo-url>
cd Facial_Emotion_Recognition_App
pip install streamlit tensorflow keras numpy pandas pillow
```

### Run the app

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (usually http://localhost:8501).

## Model

### Dataset

FER-2013: 48x48 grayscale face images in 7 emotion classes.

| Split | Images |
|---|---|
| Train | 28,709 (split 80/20 into train/validation, stratified) |
| Test | 7,178 |

The dataset is imbalanced (for example, *happy* has 7,215 training images and *disgust* only 436), so balanced class weights are used during training.

### Architecture

```
Flatten (48x48) → Dense(100, ReLU) → Dense(100, ReLU) → Dense(100, ReLU) → Dense(7, Softmax)
```

- Total parameters: 251,407
- Loss: sparse categorical cross-entropy
- Optimizer: Adam
- Epochs: 100
- Pixel values scaled to [0, 1]

### Preprocessing in the app

Each input image is converted to grayscale, resized to 48x48, and scaled to [0, 1], matching the training pipeline.

### Results (test set)

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| angry | 0.20 | 0.02 | 0.03 |
| disgust | 0.04 | 0.63 | 0.07 |
| fear | 0.17 | 0.03 | 0.05 |
| happy | 0.42 | 0.51 | 0.46 |
| neutral | 0.27 | 0.33 | 0.30 |
| sad | 0.25 | 0.02 | 0.04 |
| surprise | 0.35 | 0.54 | 0.42 |

**Overall accuracy: 26.7%** (random guessing over 7 classes is about 14%)

The model does best on *happy* and *surprise*. A dense network ignores the spatial structure of the image, which limits its accuracy.

## Possible Improvements

- Replace the dense network with a convolutional neural network (CNN)
- Add data augmentation (flips, rotations, zoom)
- Add face detection (for example, OpenCV Haar cascades) to crop the face before prediction
- Use transfer learning with a pretrained model

## Retraining the Model

1. Open `DL_lap1.ipynb` in Google Colab.
2. Run all cells. The dataset is downloaded automatically with `kagglehub`.
3. The last cell saves and downloads `emotion_model.keras`. Put it in the project folder, replacing the old one.
