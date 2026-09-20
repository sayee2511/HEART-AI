# HEART-AI 🫀

AI-powered heart disease prediction using ECG signals and deep learning.

## About

HEART-AI analyzes 12-lead ECG signals using a Convolutional Neural Network (CNN) trained on the PTB-XL ECG dataset.

The model predicts five diagnostic categories:

- NORM — Normal ECG
- MI — Myocardial Infarction
- STTC — ST/T Changes
- CD — Conduction Disturbance
- HYP — Hypertrophy

## Tech Stack

- Python
- PyTorch
- FastAPI
- React + Vite
- WFDB
- NumPy
- Scikit-learn

## Model

The project uses a 1D CNN designed for 12-lead ECG signals.

**Test Performance**
- Exact-match Accuracy: 61.12%
- Micro F1: 74.05%
- Macro F1: 65.11%

## Project Structure

HEART-AI/
├── backend/
├── frontend/
├── ml/
├── models/
├── train_ecg.py
├── evaluate_ecg.py
├── threshold_analysis.py
├── requirements.txt
└── .gitignore

## Architecture

![HEART-AI Architecture](docs/HEART-AI.png)
