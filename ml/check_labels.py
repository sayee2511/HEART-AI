import pandas as pd
import ast
from pathlib import Path

DATASET_PATH = Path(
    r"C:\Users\ADMIN\Downloads\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
)

database_file = DATASET_PATH / "ptbxl_database.csv"
scp_file = DATASET_PATH / "scp_statements.csv"

# Load files
df = pd.read_csv(database_file)
scp = pd.read_csv(scp_file, index_col=0)

print("PTB-XL LABEL ANALYSIS")
print("=" * 50)

# Convert SCP codes from text into dictionaries
df["scp_codes_dict"] = df["scp_codes"].apply(ast.literal_eval)

# Find diagnostic SCP codes
diagnostic_codes = set(
    scp[scp["diagnostic"] == 1].index
)

print("Diagnostic SCP codes found:", len(diagnostic_codes))
print()

# Extract diagnostic codes for every ECG
def get_diagnostic_codes(code_dict):
    return [
        code for code in code_dict
        if code in diagnostic_codes
    ]

df["diagnostic_codes"] = df["scp_codes_dict"].apply(
    get_diagnostic_codes
)

# Show examples
print("Example diagnostic labels:")
print()

for i in range(10):
    print(
        f"ECG {df.iloc[i]['ecg_id']}: "
        f"{df.iloc[i]['diagnostic_codes']}"
    )

print()
print("=" * 50)

# Count diagnostic codes
all_codes = []

for codes in df["diagnostic_codes"]:
    all_codes.extend(codes)

code_counts = pd.Series(all_codes).value_counts()

print("Most common diagnostic codes:")
print(code_counts.head(20))