import ast
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wfdb

from sklearn.metrics import classification_report

# -----------------------------
# PATHS
# -----------------------------

DATASET_PATH = Path(
    r"C:\Users\ADMIN\Downloads\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
)

TEST_FILE = Path("ml/test.csv")
MODEL_FILE = Path("models/ecg_cnn.pth")

LABELS = ["NORM", "MI", "STTC", "CD", "HYP"]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# -----------------------------
# CNN MODEL
# -----------------------------

class ECGCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv1d(12, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(64, 5)
        )

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x


# -----------------------------
# LOAD MODEL
# -----------------------------

model = ECGCNN().to(DEVICE)

checkpoint = torch.load(
    MODEL_FILE,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Model loaded successfully!")
print("Device:", DEVICE)


# -----------------------------
# LOAD TEST DATA
# -----------------------------

test_df = pd.read_csv(TEST_FILE)

print("Test ECGs:", len(test_df))


# -----------------------------
# PREDICTIONS
# -----------------------------

all_predictions = []
all_labels = []


with torch.no_grad():

    for index, row in test_df.iterrows():

        record_path = DATASET_PATH / row["filename_lr"]

        signal, metadata = wfdb.rdsamp(
            str(record_path)
        )

        # (1000, 12) → (12, 1000)
        signal = signal.T.astype(np.float32)

        # Normalize each lead
        mean = signal.mean(
            axis=1,
            keepdims=True
        )

        std = signal.std(
            axis=1,
            keepdims=True
        )

        signal = (
            signal - mean
        ) / (std + 1e-8)

        signal = torch.tensor(
            signal,
            dtype=torch.float32
        ).unsqueeze(0).to(DEVICE)

        # Model prediction
        output = model(signal)

        # Convert logits → probabilities
        probabilities = torch.sigmoid(output)

        # Threshold
        prediction = (
            probabilities >= 0.5
        ).cpu().numpy()[0]

        # Actual labels
        classes = ast.literal_eval(
            row["diagnostic_classes"]
        )

        actual = np.array(
            [
                1 if label in classes else 0
                for label in LABELS
            ]
        )

        all_predictions.append(prediction)
        all_labels.append(actual)

        if index % 100 == 0:

            print(
                f"Processed {index}/{len(test_df)}"
            )


# -----------------------------
# METRICS
# -----------------------------

all_predictions = np.array(
    all_predictions
)

all_labels = np.array(
    all_labels
)


print("\n==============================")
print("MODEL EVALUATION")
print("==============================")

for i, label in enumerate(LABELS):

    print(f"\n--- {label} ---")

    print(
        classification_report(
            all_labels[:, i],
            all_predictions[:, i],
            target_names=[
                "No",
                label
            ],
            zero_division=0
        )
    )