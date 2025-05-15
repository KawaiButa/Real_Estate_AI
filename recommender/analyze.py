# analyze.py

import os
import torch
import pandas as pd
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from tower import TwoTowerRec
from train_airbab import (
    PROPERTY_FIELD_LIST,
    AirbnbRecDataset,
    collate_fn,
    validate,
    generate_negative_samples,
)
property_type_list = []
DECAY_RATE = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def top_n_per_user(df: pd.DataFrame, user_col: str, score_col: str, N: int) -> pd.DataFrame:
    df_sorted = df.sort_values(
        by=[user_col, score_col],
        ascending=[True, False]
    )
    df_topn = (
        df_sorted
        .groupby(user_col, as_index=False)
        .head(N)
        .reset_index(drop=True)
    )
    return df_topn
def load_data(csv_path, grouped_reviews, number_of_negative=20):
    data = pd.read_csv(csv_path).head(8000)
    global property_type_list
    property_type_list = data["property_type"].unique().tolist()
    df = data[-1000:]
    df["property_type"] = df["property_type"].apply(
    lambda x: property_type_list.index(x)
    )
    df["price"] = df["price"].str.replace("[\$,]", "", regex=True).astype(float)
    # negative sampling omitted for brevity...
    # fill na
    for col in ["description", "reviewer_id", "score"]:
        df[col] = df[col].fillna(0 if col != "description" else "")
    df.fillna(0, inplace=True)
    df = generate_negative_samples(df, num_neg_per_user=number_of_negative)
    df = top_n_per_user(df, user_col="reviewer_id", score_col="score", N=number_of_negative + 1)
    df = df.sort_values(by="reviewer_id", ascending=True)
    dataset = AirbnbRecDataset(df, grouped_reviews)
    return DataLoader(
        dataset, batch_size=number_of_negative + 1, shuffle=False, collate_fn=collate_fn
    )


def validate_checkpoint(ckpt_path, val_loader):
    # rebuild model with the exact same args you used for training:
    search_args = {
        "text_dim": 384,
        "embed_dim": 1,
        "num_numeric": 0,
        "hidden_dim": 64,
        "out_dim": 32,
        "num_types": len(property_type_list),
        "num_cities": 0,
    }

    prop_args = {
        "struct_dim": len(PROPERTY_FIELD_LIST),
        "text_hidden_dim": 64,
        "fusion_dim": 256,
        "out_dim": 32,
    }
    model = TwoTowerRec(search_args=search_args, user_input_dim=1, prop_args=prop_args)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.to(DEVICE)

    # Wrap the val_loader with tqdm to show progress
    print("Validating...")
    val_loader = tqdm(val_loader, desc="Validating", unit="batch")

    metrics = validate(model, val_loader, k=5, threshold=0.7)
    print("Validation results:")
    for name, val in metrics.items():
        print(f"  {name}: {val:.4f}")


if __name__ == "__main__":
    import argparse
    from data import load_grouped_reviews  # your existing loader

    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True)
    args = p.parse_args()

    grouped = load_grouped_reviews("/content/reviews.csv", DECAY_RATE)
    val_loader = load_data(args.data, grouped, number_of_negative=10)
    validate_checkpoint(args.checkpoint, val_loader)
