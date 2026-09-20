from pathlib import Path
import ast

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wfdb

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"

CSV_PATH = DATASET_DIR / "ptbxl_database.csv"
SCP_PATH = DATASET_DIR / "scp_statements.csv"
MODEL_PATH = BASE_DIR / "models" / "ecg_cnn.pth"


# ============================================================
# LABELS
# ============================================================

LABELS = [
    "NORM",
    "MI",
    "STTC",
    "CD",
    "HYP"
]


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# ECG CNN
# ============================================================

class ECGCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv1d(
                12,
                32,
                kernel_size=7,
                padding=3
            ),

            nn.BatchNorm1d(32),

            nn.ReLU(),

            nn.MaxPool1d(2),

            nn.Conv1d(
                32,
                64,
                kernel_size=7,
                padding=3
            ),

            nn.BatchNorm1d(64),

            nn.ReLU(),

            nn.MaxPool1d(2),

            nn.Conv1d(
                64,
                128,
                kernel_size=5,
                padding=2
            ),

            nn.BatchNorm1d(128),

            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )


        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                128,
                64
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                64,
                5
            )
        )


    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


# ============================================================
# START
# ============================================================

print("=" * 60)
print("HEART-AI THRESHOLD ANALYSIS")
print("=" * 60)

print()

print(
    "Device:",
    DEVICE
)


# ============================================================
# LOAD MODEL
# ============================================================

model = ECGCNN().to(
    DEVICE
)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)


model.load_state_dict(
    checkpoint["model_state_dict"]
)


model.eval()


print(
    "Model loaded successfully!"
)


# ============================================================
# LOAD PTB-XL METADATA
# ============================================================

print()
print(
    "Loading PTB-XL metadata..."
)


db = pd.read_csv(
    CSV_PATH
)


scp = pd.read_csv(
    SCP_PATH,
    index_col=0
)


diagnostic_codes = set(
    scp[
        scp["diagnostic"] == 1
    ].index
)


print(
    "Total records:",
    len(db)
)


print(
    "Diagnostic SCP codes:",
    len(diagnostic_codes)
)


# ============================================================
# BUILD TEST SET
# ============================================================

print()
print(
    "Creating test labels..."
)


X_paths = []
Y = []


for _, row in db.iterrows():

    try:

        codes = ast.literal_eval(
            row["scp_codes"]
        )

    except Exception:

        continue


    labels = set()


    for code in codes.keys():

        if code not in diagnostic_codes:
            continue


        diagnostic_class = scp.loc[
            code,
            "diagnostic_class"
        ]


        if pd.isna(
            diagnostic_class
        ):

            continue


        if diagnostic_class in LABELS:

            labels.add(
                diagnostic_class
            )


    if not labels:
        continue


    # PTB-XL fold 10 = test set

    if row["strat_fold"] != 10:
        continue


    target = [

        1 if label in labels else 0

        for label in LABELS

    ]


    record_path = (
        DATASET_DIR
        / row["filename_lr"]
    )


    X_paths.append(
        record_path
    )


    Y.append(
        target
    )


print(
    "Test ECGs:",
    len(X_paths)
)


if len(X_paths) == 0:

    print()
    print(
        "ERROR: No test ECGs found."
    )

    raise SystemExit


# ============================================================
# RUN MODEL
# ============================================================

print()
print(
    "Running model predictions..."
)


y_true = []
y_prob = []


for i, record_path in enumerate(
    X_paths
):

    try:

        # ----------------------------------------
        # Read ECG
        # ----------------------------------------

        signal, metadata = wfdb.rdsamp(
            str(record_path)
        )


        # ----------------------------------------
        # Shape:
        # (1000, 12)
        # ->
        # (12, 1000)
        # ----------------------------------------

        signal = signal.T.astype(
            np.float32
        )


        # ----------------------------------------
        # Normalize each lead
        # ----------------------------------------

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
        ) / (
            std + 1e-8
        )


        # ----------------------------------------
        # Tensor
        # ----------------------------------------

        signal = torch.tensor(
            signal,
            dtype=torch.float32
        )


        signal = signal.unsqueeze(0)

        signal = signal.to(
            DEVICE
        )


        # ----------------------------------------
        # Prediction
        # ----------------------------------------

        with torch.no_grad():

            output = model(
                signal
            )

            probabilities = torch.sigmoid(
                output
            )


        probabilities = (
            probabilities
            .cpu()
            .numpy()[0]
        )


        y_true.append(
            Y[i]
        )


        y_prob.append(
            probabilities
        )


        # ----------------------------------------
        # Progress
        # ----------------------------------------

        if (i + 1) % 100 == 0:

            print(
                f"Processed "
                f"{i + 1}/"
                f"{len(X_paths)}"
            )


    except Exception as e:

        print(
            "Error processing:",
            record_path
        )

        print(
            "Reason:",
            e
        )


# ============================================================
# CONVERT TO NUMPY
# ============================================================

y_true = np.array(
    y_true
)

y_prob = np.array(
    y_prob
)


if len(y_true) == 0:

    print(
        "ERROR: No predictions completed."
    )

    raise SystemExit


# ============================================================
# THRESHOLD SEARCH
# ============================================================

print()
print("=" * 60)
print("SEARCHING FOR BEST THRESHOLDS")
print("=" * 60)

print()

# Test thresholds from 0.10 to 0.90

thresholds = np.arange(
    0.10,
    0.91,
    0.01
)


best_thresholds = []
best_f1_scores = []


for class_index, label in enumerate(
    LABELS
):

    best_threshold = 0.50
    best_f1 = -1.0
    best_precision = 0.0
    best_recall = 0.0


    true_labels = y_true[
        :,
        class_index
    ]


    probabilities = y_prob[
        :,
        class_index
    ]


    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)


        precision = precision_score(
            true_labels,
            predictions,
            zero_division=0
        )


        recall = recall_score(
            true_labels,
            predictions,
            zero_division=0
        )


        f1 = f1_score(
            true_labels,
            predictions,
            zero_division=0
        )


        if f1 > best_f1:

            best_f1 = f1

            best_threshold = (
                threshold
            )

            best_precision = (
                precision
            )

            best_recall = (
                recall
            )


    best_thresholds.append(
        best_threshold
    )

    best_f1_scores.append(
        best_f1
    )


    print(
        f"{label:5s} | "
        f"Best Threshold: "
        f"{best_threshold:.2f} | "
        f"Precision: "
        f"{best_precision:.4f} | "
        f"Recall: "
        f"{best_recall:.4f} | "
        f"F1: "
        f"{best_f1:.4f}"
    )


# ============================================================
# CURRENT 0.50 PERFORMANCE
# ============================================================

print()
print("=" * 60)
print("CURRENT THRESHOLD: 0.50")
print("=" * 60)


current_predictions = (
    y_prob >= 0.50
).astype(int)


current_macro_f1 = f1_score(
    y_true,
    current_predictions,
    average="macro",
    zero_division=0
)


current_micro_f1 = f1_score(
    y_true,
    current_predictions,
    average="micro",
    zero_division=0
)


print(
    f"Macro F1: "
    f"{current_macro_f1:.4f}"
)


print(
    f"Micro F1: "
    f"{current_micro_f1:.4f}"
)


# ============================================================
# OPTIMIZED PERFORMANCE
# ============================================================

optimized_predictions = np.zeros_like(
    y_true
)


for class_index in range(
    len(LABELS)
):

    optimized_predictions[
        :,
        class_index
    ] = (

        y_prob[
            :,
            class_index
        ]

        >=

        best_thresholds[
            class_index
        ]

    ).astype(int)


optimized_macro_f1 = f1_score(
    y_true,
    optimized_predictions,
    average="macro",
    zero_division=0
)


optimized_micro_f1 = f1_score(
    y_true,
    optimized_predictions,
    average="micro",
    zero_division=0
)


print()
print("=" * 60)
print("OPTIMIZED THRESHOLD PERFORMANCE")
print("=" * 60)


print(
    f"Macro F1: "
    f"{optimized_macro_f1:.4f}"
)


print(
    f"Micro F1: "
    f"{optimized_micro_f1:.4f}"
)


# ============================================================
# COMPARISON
# ============================================================

print()
print("=" * 60)
print("IMPROVEMENT")
print("=" * 60)


macro_change = (
    optimized_macro_f1
    - current_macro_f1
)


micro_change = (
    optimized_micro_f1
    - current_micro_f1
)


print(
    f"Macro F1 change: "
    f"{macro_change:+.4f}"
)


print(
    f"Micro F1 change: "
    f"{micro_change:+.4f}"
)


# ============================================================
# FINAL THRESHOLDS
# ============================================================

print()
print("=" * 60)
print("FINAL RECOMMENDED THRESHOLDS")
print("=" * 60)


for label, threshold in zip(
    LABELS,
    best_thresholds
):

    print(
        f"{label}: "
        f"{threshold:.2f}"
    )


# ============================================================
# FINISH
# ============================================================

print()
print("=" * 60)
print("THRESHOLD ANALYSIS COMPLETE!")
print("=" * 60)