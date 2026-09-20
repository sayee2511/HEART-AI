import ast
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import wfdb


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = Path(
    r"C:\Users\ADMIN\Downloads\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
)

TRAIN_FILE = Path("ml/train.csv")
VAL_FILE = Path("ml/validation.csv")

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

LABELS = ["NORM", "MI", "STTC", "CD", "HYP"]

BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("HEART AI - ECG CNN TRAINING")
print("=" * 60)

print("Device:", DEVICE)
print("Labels:", LABELS)


# ============================================================
# DATASET
# ============================================================

class ECGDataset(Dataset):

    def __init__(self, csv_file):

        self.df = pd.read_csv(csv_file)

        print(
            f"Loaded {len(self.df)} ECG records from {csv_file}"
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        # ----------------------------------------------------
        # ECG file path
        # ----------------------------------------------------

        record_path = DATASET_PATH / row["filename_lr"]

        # Load ECG
        signal, metadata = wfdb.rdsamp(str(record_path))

        # Shape from WFDB:
        # (1000, 12)
        #
        # PyTorch Conv1D expects:
        # (channels, samples)

        signal = signal.T

        # Convert to float32
        signal = signal.astype(np.float32)

        # ----------------------------------------------------
        # Normalize each ECG lead
        # ----------------------------------------------------

        mean = signal.mean(axis=1, keepdims=True)
        std = signal.std(axis=1, keepdims=True)

        signal = (signal - mean) / (std + 1e-8)

        # ----------------------------------------------------
        # Labels
        # ----------------------------------------------------

        classes = ast.literal_eval(
            row["diagnostic_classes"]
        )

        label = np.array(
            [
                1.0 if category in classes else 0.0
                for category in LABELS
            ],
            dtype=np.float32
        )

        signal = torch.tensor(signal)
        label = torch.tensor(label)

        return signal, label


# ============================================================
# CNN MODEL
# ============================================================

class ECGCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            # 12 ECG leads → 32 feature maps
            nn.Conv1d(
                in_channels=12,
                out_channels=32,
                kernel_size=7,
                padding=3
            ),

            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # 32 → 64
            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=7,
                padding=3
            ),

            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # 64 → 128
            nn.Conv1d(
                in_channels=64,
                out_channels=128,
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
# LOAD DATA
# ============================================================

train_dataset = ECGDataset(TRAIN_FILE)
val_dataset = ECGDataset(VAL_FILE)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# CREATE MODEL
# ============================================================

model = ECGCNN().to(DEVICE)

print()
print("CNN model created.")
print(model)


# ============================================================
# LOSS + OPTIMIZER
# ============================================================

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING
# ============================================================

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0

    print()
    print(
        f"Epoch {epoch + 1}/{EPOCHS}"
    )

    for batch_index, (signals, labels) in enumerate(train_loader):

        signals = signals.to(DEVICE)
        labels = labels.to(DEVICE)

        # Clear gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(signals)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        total_loss += loss.item()

        if batch_index % 50 == 0:

            print(
                f"Batch {batch_index}/{len(train_loader)} "
                f"Loss: {loss.item():.4f}"
            )

    average_loss = total_loss / len(train_loader)

    print(
        f"Training Loss: {average_loss:.4f}"
    )


# ============================================================
# SAVE MODEL
# ============================================================

model_path = MODEL_DIR / "ecg_cnn.pth"

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "labels": LABELS
    },
    model_path
)

print()
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print("Model saved to:")
print(model_path)