import pandas as pd

RAW_PATH = "data/raw/tickets_raw.csv"

def inspect_raw_data(path: str) -> None:
    df = pd.read_csv(path)

    print("=== 1. SHAPE & COLUMNS ===")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print("Columns:", df.columns.tolist())

    print("\n=== 2. FIRST 5 ROWS ===")
    print(df.head())

    print("\n=== 3. MISSING VALUES ===")
    print(df.isnull().sum())

    # Check duplicates on the text column (typically 'Document')
    text_col = df.columns[0]
    cat_col = df.columns[1]

    print(f"\n=== 4. DUPLICATES (on '{text_col}') ===")
    print("Exact duplicate tickets:", df.duplicated(subset=[text_col]).sum())

    print(f"\n=== 5. CATEGORY DISTRIBUTION ('{cat_col}') ===")
    counts = df[cat_col].value_counts()
    percentages = (df[cat_col].value_counts(normalize=True) * 100).round(2)
    summary = pd.DataFrame({"count": counts, "percentage (%)": percentages})
    print(summary)

if __name__ == "__main__":
    inspect_raw_data(RAW_PATH)