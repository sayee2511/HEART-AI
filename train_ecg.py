from pathlib import Path
import ast
import json

import numpy as np
import pandas as pd
import wfdb

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "models"

DATABASE_FILE = DATASET_DIR / "ptbxl_database.csv"
SCP_FILE = DATASET_DIR / "scp_statements.csv"

MODEL_PATH = MODEL_DIR / "ecg_cnn.pth"


# ============================================================
# CONFIGURATION
# ============================================================

LABELS = ["NORM", "MI", "STTC", "CD", "HYP"]

LABEL_TO_INDEX = {
    label: index
    for index, label in enumerate(LABELS)
}

BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("HEART-AI ECG TRAINING")
print("=" * 60)

print("Device:", DEVICE)


# ============================================================
# CNN MODEL
# MUST MATCH backend/predictor.py
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

            nn.Linear(128, 64),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(64, 5)
        )


    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


# ============================================================
# LOAD PTB-XL METADATA
# ============================================================

print("\nLoading PTB-XL metadata...")

database = pd.read_csv(DATABASE_FILE)

scp = pd.read_csv(
    SCP_FILE,
    index_col=0
)

print("Total ECG records:", len(database))


# ============================================================
# GET DIAGNOSTIC SCP CODES
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
# CONVERT SCP CODES → 5 DIAGNOSTIC LABELS
# ============================================================

def get_labels(scp_codes_string):

    try:

        scp_codes = ast.literal_eval(
            scp_codes_string
        )

    except Exception:

        return []


    labels = set()


    for code in scp_codes.keys():

        if code not in diagnostic_codes:
            continue


        diagnostic_class = scp.loc[
            code,
            "diagnostic_class"
        ]


        if diagnostic_class in LABEL_TO_INDEX:

            labels.add(
                diagnostic_class
            )


    return sorted(labels)


# ============================================================
# BUILD LABELS
# ============================================================

print("\nCreating diagnostic labels...")


database["labels"] = database[
    "scp_codes"
].apply(get_labels)


# Remove ECGs without diagnostic labels

database = database[
    database["labels"].map(len) > 0
].copy()


print(
    "ECGs with diagnostic labels:",
    len(database)
)


# ============================================================
# CREATE MULTI-HOT TARGETS
# ============================================================

def create_target(labels):

    target = np.zeros(
        len(LABELS),
        dtype=np.float32
    )


    for label in labels:

        target[
            LABEL_TO_INDEX[label]
        ] = 1.0


    return target


database["target"] = database[
    "labels"
].apply(create_target)


# ============================================================
# USE PTB-XL STRATIFIED FOLDS
# ============================================================

train_df = database[
    database["strat_fold"] <= 8
].copy()


validation_df = database[
    database["strat_fold"] == 9
].copy()


test_df = database[
    database["strat_fold"] == 10
].copy()


print("\nDataset split:")

print(
    "Training:",
    len(train_df)
)

print(
    "Validation:",
    len(validation_df)
)

print(
    "Test:",
    len(test_df)
)


# ============================================================
# ECG DATASET
# ============================================================

class PTBXLDataset(Dataset):

    def __init__(self, dataframe):

        self.dataframe = dataframe.reset_index(
            drop=True
        )


    def __len__(self):

        return len(self.dataframe)


    def __getitem__(self, index):

        row = self.dataframe.iloc[index]


        # ----------------------------------------------------
        # ECG path
        # ----------------------------------------------------

        record_path = (
            DATASET_DIR
            / row["filename_lr"]
        )


        # ----------------------------------------------------
        # Read ECG
        # ----------------------------------------------------

        signal, metadata = wfdb.rdsamp(
            str(record_path)
        )


        # Original:
        #
        # (1000, 12)
        #
        # Convert:
        #
        # (12, 1000)

        signal = signal.T.astype(
            np.float32
        )


        # ----------------------------------------------------
        # Normalize each lead
        # Same preprocessing as predictor.py
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Target
        # ----------------------------------------------------

        target = row["target"]


        signal = torch.tensor(
            signal,
            dtype=torch.float32
        )


        target = torch.tensor(
            target,
            dtype=torch.float32
        )


        return signal, target


# ============================================================
# DATA LOADERS
# ============================================================

train_dataset = PTBXLDataset(
    train_df
)

validation_dataset = PTBXLDataset(
    validation_df
)

test_dataset = PTBXLDataset(
    test_df
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)


validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# CREATE MODEL
# ============================================================

model = ECGCNN().to(DEVICE)


# ============================================================
# LOSS FUNCTION
# Multi-label classification
# ============================================================

criterion = nn.BCEWithLogitsLoss()


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_one_epoch():

    model.train()

    total_loss = 0.0


    for batch_index, (
        signals,
        targets
    ) in enumerate(train_loader):


        signals = signals.to(
            DEVICE
        )


        targets = targets.to(
            DEVICE
        )


        optimizer.zero_grad()


        outputs = model(
            signals
        )


        loss = criterion(
            outputs,
            targets
        )


        loss.backward()


        optimizer.step()


        total_loss += (
            loss.item()
            * signals.size(0)
        )


        if (
            batch_index + 1
        ) % 100 == 0:

            print(
                f"  Batch "
                f"{batch_index + 1}/"
                f"{len(train_loader)}"
            )


    return (
        total_loss
        / len(train_dataset)
    )


# ============================================================
# VALIDATION
# ============================================================

def validate():

    model.eval()

    total_loss = 0.0


    with torch.no_grad():

        for signals, targets in validation_loader:

            signals = signals.to(
                DEVICE
            )

            targets = targets.to(
                DEVICE
            )


            outputs = model(
                signals
            )


            loss = criterion(
                outputs,
                targets
            )


            total_loss += (
                loss.item()
                * signals.size(0)
            )


    return (
        total_loss
        / len(validation_dataset)
    )


# ============================================================
# TRAIN
# ============================================================

print("\nStarting training...\n")


best_validation_loss = float(
    "inf"
)


for epoch in range(
    1,
    EPOCHS + 1
):


    print(
        f"\nEpoch "
        f"{epoch}/{EPOCHS}"
    )


    train_loss = train_one_epoch()


    validation_loss = validate()


    print(
        f"Train Loss: "
        f"{train_loss:.4f}"
    )


    print(
        f"Validation Loss: "
        f"{validation_loss:.4f}"
    )


    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    if validation_loss < best_validation_loss:

        best_validation_loss = (
            validation_loss
        )


        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "labels":
                    LABELS,

                "validation_loss":
                    validation_loss
            },
            MODEL_PATH
        )


        print(
            "✓ Best model saved!"
        )


# ============================================================
# TEST
# ============================================================

print("\nTesting final model...")


model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )[
        "model_state_dict"
    ]
)


model.eval()


correct = 0
total = 0


with torch.no_grad():

    for signals, targets in test_loader:

        signals = signals.to(
            DEVICE
        )

        targets = targets.to(
            DEVICE
        )


        outputs = model(
            signals
        )


        probabilities = torch.sigmoid(
            outputs
        )


        predictions = (
            probabilities >= 0.5
        ).float()


        correct += (
            predictions == targets
        ).all(
            dim=1
        ).sum().item()


        total += signals.size(0)


test_accuracy = (
    correct / total
)


print(
    f"\nExact-match test accuracy: "
    f"{test_accuracy:.4f}"
)


print("\nTraining complete!")

print(
    "Model saved to:",
    MODEL_PATH
)

print("=" * 60)