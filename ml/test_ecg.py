import pandas as pd
import wfdb
from pathlib import Path

DATASET_PATH = Path(
    r"C:\Users\ADMIN\Downloads\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
)

# Load database
df = pd.read_csv(DATASET_PATH / "ptbxl_database.csv")

# Get first ECG record
row = df.iloc[0]

record_path = DATASET_PATH / row["filename_lr"]

print("ECG ID:", row["ecg_id"])
print("Record:", record_path)

# Read ECG
signal, metadata = wfdb.rdsamp(str(record_path))

print()
print("ECG loaded successfully!")
print("Signal shape:", signal.shape)
print("Sampling frequency:", metadata["fs"])
print("Number of leads:", signal.shape[1])
print("Number of samples:", signal.shape[0])

print()
print("Lead names:")
print(metadata["sig_name"])