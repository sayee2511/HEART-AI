from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import wfdb


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "ecg_cnn.pth"
)


# ============================================================
# LABELS + OPTIMIZED THRESHOLDS
# ============================================================

LABELS = [
    "NORM",
    "MI",
    "STTC",
    "CD",
    "HYP"
]


THRESHOLDS = {
    "NORM": 0.48,
    "MI": 0.21,
    "STTC": 0.42,
    "CD": 0.36,
    "HYP": 0.18
}


LABEL_NAMES = {
    "NORM": "Normal ECG",
    "MI": "Myocardial Infarction",
    "STTC": "ST/T Changes",
    "CD": "Conduction Disturbance",
    "HYP": "Hypertrophy"
}


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
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
    "ECG CNN model loaded successfully!"
)

print(
    "Device:",
    DEVICE
)


# ============================================================
# PREDICT ECG
# ============================================================

def predict_ecg(record_path):

    # --------------------------------------------------------
    # Load ECG
    # --------------------------------------------------------

    signal, metadata = wfdb.rdsamp(
        str(record_path)
    )


    # --------------------------------------------------------
    # PTB-XL shape:
    # (1000, 12)
    #
    # CNN expects:
    # (12, 1000)
    # --------------------------------------------------------

    signal = signal.T.astype(
        np.float32
    )


    # --------------------------------------------------------
    # Normalize each ECG lead
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Convert to tensor
    # --------------------------------------------------------

    signal = torch.tensor(
        signal,
        dtype=torch.float32
    )


    # Add batch dimension
    #
    # (12, 1000)
    # ->
    # (1, 12, 1000)

    signal = signal.unsqueeze(0)


    signal = signal.to(
        DEVICE
    )


    # --------------------------------------------------------
    # MODEL INFERENCE
    # --------------------------------------------------------

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


    # ========================================================
    # BUILD RESULTS
    # ========================================================

    predictions = {}

    detected_conditions = []


    for label, probability in zip(
        LABELS,
        probabilities
    ):

        probability = float(
            probability
        )


        threshold = THRESHOLDS[
            label
        ]


        detected = (
            probability >= threshold
        )


        predictions[label] = {

            "name": LABEL_NAMES[
                label
            ],

            "probability": round(
                probability,
                4
            ),

            "percentage": round(
                probability * 100,
                2
            ),

            "threshold": threshold,

            "detected": bool(
                detected
            )
        }


        if detected:

            detected_conditions.append(
                LABEL_NAMES[label]
            )


    # ========================================================
    # PRIMARY RESULT
    # ========================================================

    highest_index = int(
        np.argmax(probabilities)
    )


    primary_label = LABELS[
        highest_index
    ]


    primary_probability = float(
        probabilities[
            highest_index
        ]
    )


    primary_threshold = THRESHOLDS[
        primary_label
    ]


    primary_detected = (
        primary_probability
        >= primary_threshold
    )


    # ========================================================
    # FINAL RESPONSE
    # ========================================================

    return {

        "primary_prediction": {

            "label": primary_label,

            "name": LABEL_NAMES[
                primary_label
            ],

            "probability": round(
                primary_probability,
                4
            ),

            "percentage": round(
                primary_probability * 100,
                2
            ),

            "threshold": primary_threshold,

            "detected": bool(
                primary_detected
            )
        },


        "detected_conditions":
            detected_conditions,


        "predictions":
            predictions
    }