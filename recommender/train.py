import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from datetime import datetime
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.optim.lr_scheduler import LambdaLR

# Custom modules (make sure these are in your PYTHONPATH or project directory)
from faiss_utils import build_faiss_index
from data import (
    RecommenderDataset,
    clean_html_description,
    compute_score,
    get_search_embedding,
    PropertyEmbeddingDataset,
    sinusoidal_encode,
)
from reranker import Reranker
from tower import HierarchicalContrastiveLoss, TwoTowerRec

# ------------------------- Global Constants and Configurations -------------------------
NUM_TYPES = 6
NUM_CITIES = 20
VIETNAM_PROPERTY_CATEGORIES = [
    "apartment",
    "villa",
    "townhouse",
    "commercial",
    "land",
    "residential",
]
PROPERTY_FIELD_LIST = [
    "price",
    "bedrooms",
    "bathrooms",
    "sqm",
    "average_rating",
    "sin_lat_prop",
    "sin_lon_prop",
    "cos_lat_prop",
    "cos_lon_prop",
    "type_enc",
]
USER_FIELD_LIST = ["sin_lat", "sin_lon", "cos_lat", "cos_lon"]
tokenizer = AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_6L_768D")

# Mapping from property category to integer id
category_to_id = {cat: idx for idx, cat in enumerate(VIETNAM_PROPERTY_CATEGORIES)}

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------- Load Environment Variables -------------------------
current_file = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file)
env_path = os.path.join(parent_directory.replace("recommender", "app"), ".env")
load_dotenv(dotenv_path=env_path)


# ------------------------- Data Loading Functions -------------------------
def load_data_from_postgresql():
    engine = create_engine(os.environ.get("RECOMMENDER_DB_URL"))

    user_df = pd.read_sql(
        """
        SELECT u.id, u.name, u.email, u.min_price, u.max_price, u.phone, u.verified, u.address_id,
               a.street, a.city, a.postal_code, a.neighborhood, a.latitude, a.longitude, a.coordinates, a.geohash
        FROM users u
        LEFT JOIN addresses a ON u.address_id = a.id;
        """,
        engine,
    )

    property_df = pd.read_sql(
        """
        SELECT p.title, p.id, p.property_category, p.property_type_id, p.transaction_type, p.price,
               p.bedrooms, p.bathrooms, p.sqm, p.description, p.average_rating, p.status, p.owner_id,
               p.address_id, a.street, a.city, a.postal_code, a.neighborhood, a.latitude, a.longitude,
               a.coordinates, a.geohash
        FROM properties p
        LEFT JOIN addresses a ON p.address_id = a.id;
        """,
        engine,
    )

    actions_df = pd.read_sql(
        "SELECT user_id, property_id, action, created_at FROM user_actions", engine
    )
    search_df = pd.read_sql(
        "SELECT user_id, city, search_query, created_at, min_price, max_price, type FROM user_searches",
        engine,
    )
    user_property_df = pd.read_sql_query(
        """
        SELECT DISTINCT us.user_id, p.id
        FROM properties AS p
        JOIN user_search_properties AS usp
        ON p.id = usp.property_id
        JOIN user_search AS us
        ON usp.user_search_id = us.id
        """,
        con=engine,
    )

    # Optionally save raw data to CSV (for debugging or record-keeping)
    timestamp = str(datetime.now().timestamp())
    os.makedirs(f"data/{timestamp}", exist_ok=True)
    user_df.to_csv(f"data/{timestamp}/users.csv", index=False)
    property_df.to_csv(f"data/{timestamp}/properties.csv", index=False)
    actions_df.to_csv(f"data/{timestamp}/actions.csv", index=False)
    search_df.to_csv(f"data/{timestamp}/search.csv", index=False)
    user_property_df.to_csv(f"data/{timestamp}/search_property.csv", index=False)
    return user_df, property_df, actions_df, search_df, user_property_df


# ------------------------- Data Preprocessing Functions -------------------------
def preprocess_interactions(actions_df,search_property_df):
    # Fill missing user_id for guest actions
    
    actions_df["user_id"] = actions_df["user_id"].fillna("guest")
    actions_df["created_at"] = pd.to_datetime(actions_df["created_at"])
    actions_df["score"] = actions_df.apply(compute_score, axis=1)
    # Aggregate interactions by summing scores
    interaction_df = (
        actions_df.groupby(["user_id", "property_id"])
        .agg(score=("score", "sum"))
        .reset_index()
    )
    return interaction_df


def preprocess_property_data(property_df, tokenizer):
    # Concatenate title and clean description into a single text field
    property_df["text"] = property_df["title"] + property_df["description"].apply(
        clean_html_description
    )
    # Tokenize text (consider precomputing embeddings if needed)
    # property_df["text"] = property_df["text"].apply(
    #     lambda x: tokenizer(
    #         x,
    #         max_length=128,
    #         padding="max_length",
    #         truncation=True,
    #         return_tensors="pt",
    #     )
    # )
    # Normalize numerical columns
    cols_to_norm = ["sqm", "bedrooms", "bathrooms"]
    property_df[cols_to_norm] = (
        property_df[cols_to_norm] - property_df[cols_to_norm].mean()
    ) / property_df[cols_to_norm].std()
    # Process latitude/longitude with sinusoidal encoding
    property_df["lat"] = pd.to_numeric(property_df["latitude"], errors="coerce").fillna(
        0
    )
    property_df["lon"] = pd.to_numeric(
        property_df["longitude"], errors="coerce"
    ).fillna(0)
    property_df = pd.concat(
        [property_df, property_df.apply(sinusoidal_encode, axis=1)], axis=1
    )
    # Encode property category
    property_df["type_enc"] = property_df["property_category"].apply(
        lambda x: category_to_id[x]
    )
    return property_df


def preprocess_user_data(user_df):
    user_df["lat"] = pd.to_numeric(user_df["latitude"], errors="coerce").fillna(0)
    user_df["lon"] = pd.to_numeric(user_df["longitude"], errors="coerce").fillna(0)
    user_df = pd.concat([user_df, user_df.apply(sinusoidal_encode, axis=1)], axis=1)
    return user_df


def merge_data(interaction_df, user_df, property_df):
    df = interaction_df.merge(
        user_df, left_on="user_id", right_on="id", how="left", suffixes=("", "_user")
    )
    df = df.merge(
        property_df,
        left_on="property_id",
        right_on="id",
        how="left",
        suffixes=("", "_prop"),
    )
    # Fill defaults for guest/missing user data
    default_lat = user_df["lat"].median()
    default_lon = user_df["lon"].median()
    default_min_price = user_df["min_price"].median()
    default_max_price = user_df["max_price"].median()
    df["min_price"].fillna(default_min_price, inplace=True)
    df["max_price"].fillna(default_max_price, inplace=True)
    df["lat"].fillna(default_lat, inplace=True)
    df["lon"].fillna(default_lon, inplace=True)
    return df


def preprocess_search_data(search_df):
    if search_df.empty:
        return None
    search_df["text_emb"] = search_df["search_query"].apply(get_search_embedding)
    search_df["type"] = (
        search_df["type"].apply(lambda x: category_to_id[x] if x else -1) + 1
    )
    search_df["created_at"] = pd.to_datetime(search_df["created_at"])
    search_df["range_price"] = search_df["max_price"] - search_df["min_price"]
    search_df["average_price"] = search_df["range_price"] / 2
    search_df.fillna(0, inplace=True)
    # Group by user and pad search records
    search_df["user_id"] = search_df["user_id"].astype(str)
    search_grouped = search_df.groupby("user_id")
    text_dim = 384
    num_numeric = 4
    M = text_dim + 1 + 1 + num_numeric
    N = 5
    B = len(search_df)  # alternatively, number of users you expect in merged data
    result = np.zeros((B, N, M), dtype=np.float32)
    for i, (user_id, group) in enumerate(search_grouped):
        matches = group.index
        group = group.sort_values(by="created_at", ascending=False).head(N)
        padded = np.zeros((N, M), dtype=np.float32)
        for j, row in enumerate(group.itertuples()):
            vec = np.concatenate(
                [
                    row.text_emb,
                    [row.type],
                    [row.city],
                    [row.min_price],
                    [row.max_price],
                    [row.average_price],
                    [row.range_price],
                ]
            )
            padded[j] = vec
        result[i] = padded
    return result


# ------------------------- Main Data Loading and Preprocessing -------------------------
def main_data_pipeline():
    # Load raw data from PostgreSQL
    user_df, property_df, actions_df, search_df, search_property_df = load_data_from_postgresql()

    # Preprocess each DataFrame
    interaction_df = preprocess_interactions(actions_df)
    property_df = preprocess_property_data(property_df, tokenizer)
    user_df = preprocess_user_data(user_df)
    merged_df = merge_data(interaction_df, user_df, property_df)
    search_feats = preprocess_search_data(search_df)
    

    # Prepare features for model training
    user_features = merged_df[USER_FIELD_LIST].values.astype(np.float32)
    property_feats = merged_df[PROPERTY_FIELD_LIST].values.astype(np.float32)
    property_text_feats = merged_df["text"].values
    ratings = merged_df["score"].values.astype(np.float32)
    # Normalize ratings to [0,1] (Sigmoid)
    ratings = 1 / (1 + np.exp(-ratings))

    return (
        user_features,
        property_feats,
        search_feats,
        property_text_feats,
        ratings,
        merged_df,
    )


user_features, property_feats, search_feats, property_text_feats, ratings, merged_df = (
    main_data_pipeline()
)

# Create Dataset and DataLoader
dataset = RecommenderDataset(
    user_features, property_feats, search_feats, property_text_feats, ratings, tokenizer
)
train_loader = DataLoader(dataset, batch_size=128, shuffle=True)

# ------------------------- Model and Loss Initialization -------------------------
two_tower = TwoTowerRec(
    search_text_dim=384,
    search_embed_dim=1,
    search_num_numeric=4,
    search_out_dim=32,
    search_num_cities=NUM_CITIES,
    search_num_types=NUM_TYPES,
    user_input_dim=32 + len(USER_FIELD_LIST),
    property_input_dim=len(PROPERTY_FIELD_LIST),
)
optimizer = torch.optim.AdamW(
    [
        {"params": two_tower.parameters(), "lr": 1e-4},
        {"params": two_tower.view_projectors.parameters(), "lr": 3e-4},
    ]
)
num_training_steps = 10 * len(train_loader)
num_warmup_steps = int(0.1 * num_training_steps)


def lr_lambda(current_step):
    warmup_steps = num_warmup_steps
    total_steps = num_training_steps
    if current_step < warmup_steps:
        # linear warmup
        return float(current_step) / float(max(1, warmup_steps))
    # linear decay
    return max(
        0.0,
        float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)),
    )


scheduler = LambdaLR(optimizer, lr_lambda)
loss_fn = HierarchicalContrastiveLoss(margin=1.0, temp=0.5)


# ------------------------- Training Functions -------------------------
def train_two_tower(model, loader, optimizer, loss_fn, epochs=10, device=DEVICE):
    model.to(device)
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            user = batch["user"]["data"].to(device)  # [B, M]
            search = batch["user"]["searches"].to(device)  # [B, N, M]
            prop = batch["property"]["data"].to(device)  # [B, M]
            prop_text = batch["property"]["text"]
            targets = batch["target"].to(device).float().unsqueeze(1)  # [B, 1]
            u_emb, p_emb, p_view = model(user, prop, search, prop_text)
            loss = loss_fn(u_emb, p_emb, p_view, targets)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {total_loss / len(loader):.4f}")
    torch.save(model.state_dict(), "two_tower_rec.pth")


def train_reranker(reranker, dataloader, model, epochs=10, device=DEVICE):
    reranker.to(device)
    model.to(device)
    optimizer_rerank = optim.Adam(reranker.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    reranker.train()
    model.eval()
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Reranker Epoch {epoch+1}/{epochs}"):
            user_feats = batch["user"]["data"].to(device)
            search_feats = batch["user"]["searches"].to(device)
            prop_feats = batch["property"]["data"].to(device)
            prop_text_feats = batch["property"]["text"]
            targets = batch["target"].to(device).float().unsqueeze(1)
            with torch.no_grad():
                user_emb, prop_emb, _ = model(
                    user_feats, prop_feats, search_feats, prop_text_feats
                )
            reranker_input = torch.cat([user_emb, prop_emb], dim=-1)
            preds = reranker(reranker_input)
            loss = criterion(preds, targets)
            optimizer_rerank.zero_grad()
            loss.backward()
            optimizer_rerank.step()
            total_loss += loss.item()
        print(f"Reranker Epoch {epoch+1} Loss: {total_loss / len(dataloader):.4f}")
    torch.save(reranker.state_dict(), "reranker.pth")


# ------------------------- Main Training Script -------------------------
if __name__ == "__main__":
    # Train Two-Tower Model
    train_two_tower(two_tower, train_loader, optimizer, loss_fn, epochs=10)

    # Load pre-trained state (optional, if continuing from a saved model)
    # two_tower_state_dict = torch.load(
    #     "two_tower_rec.pth", map_location=torch.device("cpu")
    # )
    # two_tower.load_state_dict(two_tower_state_dict)

    # Build FAISS index for property retrieval
    property_data = PropertyEmbeddingDataset(property_feats, property_text_feats)
    build_faiss_index(two_tower, merged_df)

    # Train reranker model
    reranker = Reranker()
    train_reranker(reranker, train_loader, two_tower, epochs=10)

    # Save final property features
    merged_df.to_csv("property_features.csv", index=False)
