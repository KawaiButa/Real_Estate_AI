import os
import json
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

from tower import TwoTowerRec
from data import load_grouped_reviews
from transformers import AutoTokenizer

PROPERTY_FIELD_LIST = [
    "latitude",
    "longitude",
    "accommodates",
    "bathrooms",
    "bedrooms",
    "price",
    "property_type",
]
MAX_REVIEWS = 5
DECAY_RATE = 0.001
MAX_LEN = 512

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PropertyDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        grouped_reviews: dict,
        tokenizer_name: str,
    ):
        super().__init__()
        self.df = df
        self.review_map = grouped_reviews
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        listing_id = row["id_x"]

        # structured features
        struct_vals = row[PROPERTY_FIELD_LIST].values.astype(float)
        struct_feat = torch.tensor(struct_vals, dtype=torch.float)
        reviews_list = self.review_map.get(listing_id, [])
        reviews_list = sorted(reviews_list, key=lambda x: -x[1])[:MAX_REVIEWS]

        texts = [x[0] for x in reviews_list]
        weights = [x[1] for x in reviews_list]
        R = len(texts)
        # tokenize all texts
        encodings = (
            self.tokenizer(
                texts,
                max_length=MAX_LEN,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            if R > 0
            else {
                "input_ids": torch.zeros((0, MAX_LEN), dtype=torch.long),
                "attention_mask": torch.zeros((0, MAX_LEN), dtype=torch.long),
            }
        )
        # tokenize description
        text_tok = self.tokenizer(
            row["description"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        # squeeze off batch dim
        text_seq = {
            "input_ids": text_tok["input_ids"].squeeze(0),
            "attention_mask": text_tok["attention_mask"].squeeze(0),
        }

        return {
            "listing_id": listing_id,
            "struct_feat": struct_feat,
            "text_seq": text_seq,
            "review_seq": encodings,
        }


def load_full_data(csv_path, grouped_reviews):
    df = pd.read_csv(csv_path).head(100)
    property_type_list = df["property_type"].unique().tolist()
    df["property_type"] = df["property_type"].apply(
        lambda x: property_type_list.index(x)
    )
    df["price"] = df["price"].str.replace("[\$,]", "", regex=True).astype(float)

    for col in ["description", "reviewer_id", "score"]:
        df[col] = df[col].fillna(0 if col != "description" else "")
    df.fillna(0, inplace=True)

    dataset = PropertyDataset(
        df, grouped_reviews, "huawei-noah/TinyBERT_General_6L_768D"
    )
    loader = DataLoader(
        dataset, batch_size=64, shuffle=False, collate_fn=property_collate_fn
    )
    return loader, len(property_type_list)


def property_collate_fn(batch):
    """
    Collate a list of property‐samples (from PropertyDataset) into a single batch.
    Pads review_seq out to the maximum number of reviews in the batch.
    """
    # listing_ids
    listing_ids = [item["listing_id"] for item in batch]

    # stack struct_feats
    struct_feats = torch.stack([item["struct_feat"] for item in batch], dim=0)

    # stack text_seq
    input_ids = torch.stack([item["text_seq"]["input_ids"] for item in batch], dim=0)
    attention = torch.stack(
        [item["text_seq"]["attention_mask"] for item in batch], dim=0
    )
    text_seq = {"input_ids": input_ids, "attention_mask": attention}

    # pad reviews to max R
    Rs = [item["review_seq"]["input_ids"].size(0) for item in batch]
    max_R = max(Rs)
    padded_ids, padded_masks = [], []
    for item in batch:
        ids = item["review_seq"]["input_ids"]
        mask = item["review_seq"]["attention_mask"]
        pad = max_R - ids.size(0)
        if pad > 0:
            ids = torch.cat(
                [ids, torch.zeros((pad, ids.size(1)), dtype=ids.dtype)], dim=0
            )
            mask = torch.cat(
                [mask, torch.zeros((pad, mask.size(1)), dtype=mask.dtype)], dim=0
            )
        padded_ids.append(ids)
        padded_masks.append(mask)

    review_seq = {
        "input_ids": torch.stack(padded_ids, dim=0),  # [B, max_R, L]
        "attention_mask": torch.stack(padded_masks, dim=0),
    }

    return {
        "listing_id": listing_ids,
        "struct_feat": struct_feats,
        "text_seq": text_seq,
        "review_seq": review_seq,
    }


def infer_embeddings(model: TwoTowerRec, loader: DataLoader):
    model.eval()
    prop_ids, prop_vecs = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inferring embeddings"):
            prop_ids_batch = batch["listing_id"]
            struct = batch["struct_feat"].to(DEVICE)
            text = {k: v.to(DEVICE) for k, v in batch["text_seq"].items()}
            reviews = {k: v.to(DEVICE) for k, v in batch["review_seq"].items()}
            prop_emb, _ = model.prop_tower(struct, text, reviews)
            prop_ids.extend(prop_ids_batch)
            prop_vecs.extend(prop_emb.tolist())
    return prop_ids, prop_vecs

def save_embeddings(path, ids, embs):
    """
    Save embeddings to a JSON file as a list of {"id": int, "embedding": [floats...]}.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    records = []
    for idx, emb in zip(ids, embs):
        # If idx is a torch.Tensor or numpy scalar:
        py_id = int(idx)  

        # If emb is a torch.Tensor:
        if hasattr(emb, "cpu"):
            emb = emb.cpu().detach()
        # Now emb should be a numpy array or Tensor; convert to list
        emb_list = list(emb)

        records.append({
            "id": py_id,
            "embedding": emb_list
        })

    with open(path, "w") as f:
        json.dump(records, f)
    print(f"Saved {len(records)} embeddings to {path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output_dir", default=".")
    args = parser.parse_args()

    grouped_reviews = load_grouped_reviews("/content/reviews.csv", DECAY_RATE)
    loader, num_property_types = load_full_data(args.data, grouped_reviews)

    # Rebuild model (same args as in analyze.py)
    search_args = {
        "text_dim": 384,
        "embed_dim": 1,
        "num_numeric": 0,
        "hidden_dim": 64,
        "out_dim": 32,
        "num_types": 18,
        "num_cities": 0,
    }

    prop_args = {
        "struct_dim": len(PROPERTY_FIELD_LIST),
        "text_hidden_dim": 64,
        "fusion_dim": 256,
        "out_dim": 32,
    }

    model = TwoTowerRec(search_args=search_args, user_input_dim=1, prop_args=prop_args)
    model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
    model.to(DEVICE)

    # Inference
    prop_ids, prop_embs = infer_embeddings(model, loader)

    os.makedirs(args.output_dir, exist_ok=True)
    save_embeddings(
        os.path.join(args.output_dir, "property_embeddings.json"), prop_ids, prop_embs
    )

    print("Embeddings saved to:", args.output_dir)
