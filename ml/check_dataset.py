import pandas as pd
from pathlib import Path
import ast

# PTB-XL dataset location
DATASET_PATH = Path(
    r"C:\Users\ADMIN\Downloads\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
)

# Files
database_file = DATASET_PATH / "ptbxl_database.csv"
scp_file = DATASET_PATH / "scp_statements.csv"

print("Checking PTB-XL dataset...")
print()

# Check files
print("Database file exists:", database_file.exists())
print("SCP file exists:", scp_file.exists())
print("records100 exists:", (DATASET_PATH / "records100").exists())
print("records500 exists:", (DATASET_PATH / "records500").exists())

print()

# Load database
df = pd.read_csv(database_file)

print("Number of ECG records:", len(df))
print("Number of columns:", len(df.columns))

print()
print("Important columns:")
print(df[["ecg_id", "patient_id", "scp_codes", "filename_lr", "filename_hr", "strat_fold"]].head())

print()
print("Dataset loaded successfully!")