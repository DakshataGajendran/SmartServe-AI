import json
import pandas as pd
from sklearn.model_selection import train_test_split
from preprocess import clean_ticket_text

RAW_PATH = "data/raw/tickets_raw.csv"
TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"
REF_DIST_PATH = "data/processed/reference_distribution.json"

def prepare_dataset() -> None:
    df = pd.read_csv(RAW_PATH)

    # 1. Standardize schema to match blueprint
    df = df.rename(columns={"Document": "ticket_text", "Topic_group": "category"})
    df["ticket_text"] = df["ticket_text"].apply(clean_ticket_text)
    df["category"] = df["category"].str.strip()

    # 2. Drop empty tickets or duplicates after cleaning
    df = df[df["ticket_text"].str.len() > 0].drop_duplicates(subset=["ticket_text"]).reset_index(drop=True)
    df.insert(0, "ticket_id", range(1, len(df) + 1))

    # 3. Stratified 80/20 Train/Test Split (prevents data leakage)
    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df["category"]
    )

    # 4. Save training reference distribution for drift monitoring
    ref_dist = (train_df["category"].value_counts(normalize=True)).round(4).to_dict()
    with open(REF_DIST_PATH, "w", encoding="utf-8") as f:
        json.dump(ref_dist, f, indent=2)

    # 5. Save processed splits
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    print(f"Total clean rows : {len(df)}")
    print(f"Train set rows   : {len(train_df)} -> saved to {TRAIN_PATH}")
    print(f"Test set rows    : {len(test_df)} -> saved to {TEST_PATH}")
    print(f"Reference dist   : saved to {REF_DIST_PATH}")

if __name__ == "__main__":
    prepare_dataset()