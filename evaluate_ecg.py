from pathlib import Path
import ast

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wfdb

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
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


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# ECG CNN MODEL
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
print("HEART-AI MODEL EVALUATION")
print("=" * 60)

print()

print("Device:", DEVICE)


# ============================================================
# LOAD MODEL
# ============================================================

model = ECGCNN().to(DEVICE)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)


model.load_state_dict(
    checkpoint["model_state_dict"]
)


model.eval()


print("Model loaded successfully!")


# ============================================================
# LOAD DATASET
# ============================================================

print()
print("Loading PTB-XL metadata...")


db = pd.read_csv(
    CSV_PATH
)


# IMPORTANT:
# The SCP code is the first column/index.
# We load it as the DataFrame index.

scp = pd.read_csv(
    SCP_PATH,
    index_col=0
)


print(
    "Total records:",
    len(db)
)


# ============================================================
# FIND DIAGNOSTIC SCP CODES
# ============================================================

diagnostic_codes = set(
    scp[
        scp["diagnostic"] == 1
    ].index
)


print(
    "Diagnostic SCP codes:",
    len(diagnostic_codes)
)


# ============================================================
# CREATE TEST LABELS
# ============================================================

print()
print("Creating test labels...")


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


    # Ignore ECGs without
    # diagnostic labels.

    if not labels:
        continue


    # PTB-XL fold 10
    # is our test set.

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


# ============================================================
# SAFETY CHECK
# ============================================================

if len(X_paths) == 0:

    print()
    print("=" * 60)
    print("ERROR")
    print("=" * 60)

    print(
        "No test ECGs were found."
    )

    print(
        "Please check the PTB-XL metadata and label mapping."
    )

    raise SystemExit


# ============================================================
# RUN PREDICTIONS
# ============================================================

print()
print("Running predictions...")


y_true = []
y_pred = []
y_prob = []


for i, record_path in enumerate(
    X_paths
):

    try:

        # ----------------------------------------
        # Load ECG
        # ----------------------------------------

        signal, metadata = wfdb.rdsamp(
            str(record_path)
        )


        # ----------------------------------------
        # Convert:
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
        # Convert to Tensor
        # ----------------------------------------

        signal = torch.tensor(
            signal,
            dtype=torch.float32
        )


        # Add batch dimension:
        # (12, 1000)
        # ->
        # (1, 12, 1000)

        signal = signal.unsqueeze(0)


        signal = signal.to(
            DEVICE
        )


        # ----------------------------------------
        # Model prediction
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


        # ----------------------------------------
        # Convert probabilities
        # to binary predictions
        # ----------------------------------------

        predictions = (
            probabilities >= 0.5
        ).astype(int)


        y_true.append(
            Y[i]
        )


        y_pred.append(
            predictions
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
# CONVERT RESULTS TO NUMPY
# ============================================================

y_true = np.array(
    y_true
)

y_pred = np.array(
    y_pred
)

y_prob = np.array(
    y_prob
)


# ============================================================
# FINAL SAFETY CHECK
# ============================================================

if len(y_true) == 0:

    print()
    print(
        "No ECG predictions were successfully completed."
    )

    raise SystemExit


# ============================================================
# EVALUATION
# ============================================================

print()
print()
print("=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)


# Exact match

exact_accuracy = accuracy_score(
    y_true,
    y_pred
)


# Micro metrics

micro_precision = precision_score(
    y_true,
    y_pred,
    average="micro",
    zero_division=0
)


micro_recall = recall_score(
    y_true,
    y_pred,
    average="micro",
    zero_division=0
)


micro_f1 = f1_score(
    y_true,
    y_pred,
    average="micro",
    zero_division=0
)


# Macro metrics

macro_precision = precision_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)


macro_recall = recall_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)


macro_f1 = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)


# ============================================================
# PRINT OVERALL METRICS
# ============================================================

print()

print(
    f"Exact-match Accuracy : "
    f"{exact_accuracy:.4f}"
)


print(
    f"Micro Precision      : "
    f"{micro_precision:.4f}"
)


print(
    f"Micro Recall         : "
    f"{micro_recall:.4f}"
)


print(
    f"Micro F1             : "
    f"{micro_f1:.4f}"
)


print(
    f"Macro Precision      : "
    f"{macro_precision:.4f}"
)


print(
    f"Macro Recall         : "
    f"{macro_recall:.4f}"
)


print(
    f"Macro F1             : "
    f"{macro_f1:.4f}"
)


# ============================================================
# PER-CLASS PERFORMANCE
# ============================================================

print()
print("=" * 60)
print("PER-CLASS PERFORMANCE")
print("=" * 60)


report = classification_report(

    y_true,

    y_pred,

    target_names=LABELS,

    zero_division=0
)


print(
    report
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("=" * 60)
print("TEST SET CLASS DISTRIBUTION")
print("=" * 60)


for i, label in enumerate(
    LABELS
):

    count = int(
        y_true[:, i].sum()
    )


    total = len(
        y_true
    )


    percentage = (
        count / total
    ) * 100


    print(
        f"{label:5s}: "
        f"{count:4d} "
        f"({percentage:.2f}%)"
    )


# ============================================================
# FINISH
# ============================================================

print()
print("=" * 60)
print("EVALUATION COMPLETE!")
print("=" * 60)