import pandas as pd
import ast
from pathlib import Path

# =========================
# PTB-XL DATASET PATH
# =========================

DATASET_PATH = Path(
    r"C:\Users\ADMIN\Downloads\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
)

DATABASE_FILE = DATASET_PATH / "ptbxl_database.csv"
SCP_FILE = DATASET_PATH / "scp_statements.csv"


# =========================
# LOAD DATA
# =========================

df = pd.read_csv(DATABASE_FILE)
scp = pd.read_csv(SCP_FILE, index_col=0)

print("PTB-XL loaded:", len(df), "ECGs")


# =========================
# CONVERT SCP CODES
# =========================

df["scp_codes_dict"] = df["scp_codes"].apply(ast.literal_eval)


# =========================
# GET DIAGNOSTIC CODES
# =========================

diagnostic_scp = scp[scp["diagnostic"] == 1]

print("Diagnostic SCP codes:", len(diagnostic_scp))


# =========================
# MAP TO SUPERCLASSES
# =========================

def get_superclasses(code_dict):

    classes = []

    for code in code_dict:

        if code in diagnostic_scp.index:

            superclass = diagnostic_scp.loc[code, "diagnostic_class"]

            if pd.notna(superclass):
                classes.append(superclass)

    return list(set(classes))


df["diagnostic_classes"] = df["scp_codes_dict"].apply(
    get_superclasses
)


# =========================
# SHOW EXAMPLES
# =========================

print("\nExample labels:\n")

for i in range(10):

    print(
        f"ECG {df.iloc[i]['ecg_id']} → "
        f"{df.iloc[i]['diagnostic_classes']}"
    )


# =========================
# CLASS COUNTS
# =========================

print("\nDiagnostic superclass counts:")

class_counts = {}

for classes in df["diagnostic_classes"]:

    for c in classes:

        class_counts[c] = class_counts.get(c, 0) + 1


for c, count in sorted(
    class_counts.items(),
    key=lambda x: x[1],
    reverse=True
):

    print(f"{c}: {count}")


# =========================
# TRAIN / VALIDATION / TEST
# =========================

train_df = df[df["strat_fold"].between(1, 8)]

validation_df = df[df["strat_fold"] == 9]

test_df = df[df["strat_fold"] == 10]


print("\nDataset split:")

print("Training:", len(train_df))
print("Validation:", len(validation_df))
print("Testing:", len(test_df))


# =========================
# SAVE PREPARED DATA
# =========================

OUTPUT_DIR = Path("ml")

train_df.to_csv(
    OUTPUT_DIR / "train.csv",
    index=False
)

validation_df.to_csv(
    OUTPUT_DIR / "validation.csv",
    index=False
)

test_df.to_csv(
    OUTPUT_DIR / "test.csv",
    index=False
)


print("\nPrepared datasets saved!")
print("ml/train.csv")
print("ml/validation.csv")
print("ml/test.csv")