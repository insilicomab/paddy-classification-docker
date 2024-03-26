import pandas as pd


def generate_label_map(df: pd.DataFrame) -> dict:
    unique_labels = df["label"].unique()
    label_map = {label: int for int, label in enumerate(unique_labels)}
    return label_map


def load_df_from_csv(file_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(file_path)
    train_df = df[df["is_valid"] == 0]
    val_df = df[df["is_valid"] == 1]
    return train_df, val_df
