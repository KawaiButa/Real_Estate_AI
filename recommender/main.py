import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import faiss
from sqlalchemy import create_engine
from datetime import datetime
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from recommender.data import RecommenderDataset, get_search_embedding
from recommender.faiss import build_faiss_index
from recommender.reranker import Reranker, build_rerank_data
from recommender.tower import TwoTowerRec, cosine_loss
load_dotenv()
property_field_list = ["price", "average_rating", "lat_prop", "lon_prop", "type_enc"]
user_field_list = ["age", "min_price", "max_price", "lat", "lon"]
# -------------------- Step 1: Load Data from PostgreSQL --------------------
engine = create_engine(os.environ.get("DB_URL"))
user_df = pd.read_sql(
    "SELECT u.id, u.name, u.email, u.phone, u.verified, u.address_id, a.street, a.city, a.postal_code, a.neighborhood, a.latitude, a.longitude, a.coordinates, a.geohash FROM users u LEFT JOIN addresses a ON u.address_id = a.id;",
    engine,
)
property_df = pd.read_sql(
    "SELECT p.title, p.id, p.property_category, p.property_type_id, p.transaction_type, p.price, p.bedrooms, p.bathrooms, p.sqm, p.description, p.average_rating, p.status, p.owner_id, p.address_id, a.street, a.city, a.postal_code, a.neighborhood, a.latitude, a.longitude, a.coordinates, a.geohash FROM properties p LEFT JOIN addresses a ON p.address_id = a.id;",
    engine,
)
actions_df = pd.read_sql(
    "SELECT user_id, property_id, action, created_at FROM user_actions", engine
)
search_df = pd.read_sql(
    "SELECT user_id, search_query, created_at, min_price, max_price, property_type FROM user_searches",
    engine,
)

# -------------------- Preprocess Interactions --------------------
# Guest actions: fill NaN user_id with "guest"
actions_df["user_id"] = actions_df["user_id"].fillna("guest")


def compute_score(row):
    decay = np.exp(-((datetime.now() - row["created_at"]).days) / 30)
    if row["action"] == "like":
        return 3 * decay
    elif row["action"] == "view":
        return 1 * decay
    elif row["action"] == "unlike":
        return -2 * decay
    return 0


actions_df["created_at"] = pd.to_datetime(actions_df["created_at"])
actions_df["score"] = actions_df.apply(compute_score, axis=1)
interaction_df = (
    actions_df.groupby(["user_id", "property_id"])
    .agg(score=("score", "sum"))
    .reset_index()
)

# -------------------- Process Search History with SentenceTransformer --------------------

search_df["created_at"] = pd.to_datetime(search_df["created_at"])
search_embeddings = search_df["search_query"].apply(get_search_embedding)
search_emb_df = pd.DataFrame(search_embeddings.tolist(), index=search_df.index)
search_df = pd.concat([search_df, search_emb_df], axis=1)
search_emb_cols = [f"s{i}" for i in range(16)]
search_df_cols = list(search_df.columns[:-16]) + search_emb_cols
search_df.columns = search_df_cols
search_agg = (
    search_df.groupby("user_id")
    .agg(
        lambda x: (
            np.mean(np.stack(x), axis=0)
            if x.dtype == "O" or isinstance(x.iloc[0], np.ndarray)
            else x.mean()
        )
    )
    .reset_index()
)

# Use Latitude/Longitude for Location
user_df["lat"] = pd.to_numeric(user_df["latitude"], errors="coerce").fillna(0)
user_df["lon"] = pd.to_numeric(user_df["longitude"], errors="coerce").fillna(0)
user_df["min_price"] = pd.to_numeric(user_df["min_price"], errors="coerce").fillna(0)
user_df["max_price"] = pd.to_numeric(user_df["max_price"], errors="coerce").fillna(0)
property_df["lat"] = pd.to_numeric(property_df["latitude"], errors="coerce").fillna(0)
property_df["lon"] = pd.to_numeric(property_df["longitude"], errors="coerce").fillna(0)
# Keep property_category for property type filter
# (We already have property_category; use that as type, no encoding on it if preferred)
# Alternatively, encode it:
property_df["type_enc"] = property_df["property_category"].astype("category").cat.codes

#  Merge Data for Training
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
df = df.merge(
    search_agg[["user_id"] + search_emb_cols], on="user_id", how="left"
).fillna(0)
# For guest actions missing user data, fill defaults.
default_age = user_df["age"].mean()
default_lat = user_df["lat"].median()
default_lon = user_df["lon"].median()
default_min_price = user_df["min_price"].median()
default_max_price = user_df["max_price"].median()
df["age"].fillna(default_age, inplace=True)
df["min_price"].fillna(default_min_price, inplace=True)
df["max_price"].fillna(default_max_price, inplace=True)
df["lat"].fillna(default_lat, inplace=True)
df["lon"].fillna(default_lon, inplace=True)

# Feature Setup
# User features: [age, max_price, min_price, lat, lon] + search embedding (16 dims) = 5+16 = 21 dims
user_feats = df[user_field_list].values.astype(
    np.float32
)
search_feats = df[search_emb_cols].values.astype(np.float32)
user_features = np.concatenate([user_feats, search_feats], axis=1)

# Property features: [price, rating, lat, lon, type_enc] = 6 dims
property_feats = df[property_field_list].values.astype(np.float32)
ratings = df["score"].values.astype(np.float32)

dataset = RecommenderDataset(user_features, property_feats, ratings)
train_loader = DataLoader(dataset, batch_size=256, shuffle=True)


two_tower = TwoTowerRec()
optimizer = optim.Adagrad(two_tower.parameters(), lr=0.1)
for epoch in range(5):
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        user = torch.tensor(batch["user"], dtype=torch.float32)
        prop = torch.tensor(batch["property"], dtype=torch.float32)
        u_emb, p_emb = two_tower(user, prop)
        loss = cosine_loss(u_emb, p_emb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")
torch.save(two_tower.state_dict(), "two_tower_rec.pth")

build_faiss_index(two_tower, property_df)


# Prepare data for train reranker
X_rerank, y_rerank = build_rerank_data(two_tower, df)
X_train, X_val, y_train, y_val = train_test_split(X_rerank, y_rerank, test_size=0.2)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

# Train reranker
reranker = Reranker()
criterion = nn.MSELoss()
opt_rerank = optim.Adam(reranker.parameters(), lr=0.001)
for epoch in range(5):
    reranker.train()
    opt_rerank.zero_grad()
    preds = reranker(X_train_tensor)
    loss = criterion(preds, y_train_tensor)
    loss.backward()
    opt_rerank.step()
    reranker.eval()
    with torch.no_grad():
        val_loss = criterion(reranker(X_val_tensor), y_val_tensor)
    print(
        f"Reranker Epoch {epoch+1}: Loss {loss.item():.4f}, Val Loss {val_loss.item():.4f}"
    )
torch.save(reranker.state_dict(), "reranker.pth")


# -------------------- Inference: Retrieval + Reranking --------------------
def retrieve_candidates(user_vector, top_k=20):
    user_tensor = torch.tensor(user_vector, dtype=torch.float32)
    with torch.no_grad():
        u_emb = two_tower.user_tower(user_tensor).numpy()
    faiss.normalize_L2(u_emb)
    index = faiss.read_index("property_faiss.index")
    ids = np.load("property_id_map.npy")
    dists, idxs = index.search(u_emb, top_k)
    return ids[idxs[0]], u_emb


def recommend(
    two_tower, reranker, user_raw_vector, property_type_filter=None, boost=0.1
):
    user_raw = np.array([user_raw_vector], dtype=np.float32)
    candidate_ids, u_emb = retrieve_candidates(two_tower, user_raw, top_k=20)
    results = []
    for pid in candidate_ids:
        prop = property_df[property_df["id"] == pid]
        p_vec = prop[property_field_list].values.astype(np.float32)
        p_tensor = torch.tensor(p_vec, dtype=torch.float32)
        with torch.no_grad():
            p_emb = two_tower.prop_tower(p_tensor).numpy()
        combined = np.concatenate([u_emb, p_emb], axis=1)
        combined_tensor = torch.tensor(combined, dtype=torch.float32)
        with torch.no_grad():
            score = reranker(combined_tensor).numpy()[0][0]
        # Boost score if property type matches filter
        if property_type_filter is not None:
            prop_type = prop["property_category"].iloc[0]
            if prop_type == property_type_filter:
                score += boost
        results.append((pid, score))
    results.sort(key=lambda x: -x[1])
    return [pid for pid, _ in results]


# -------------------- Example Inference Usage --------------------
# For a user with: age=30, income=70000, lat and lon from address,
# plus aggregated search embedding (if no search history, zeros)
example_user_features = np.array([30, 40.7128, -74.0060] + [0] * 16, dtype=np.float32)
# If a property type filter is desired, e.g., "Apartment"
recommended_properties = recommend(
    two_tower, reranker, example_user_features, property_type_filter="Apartment"
)
print("Recommended property IDs:", recommended_properties)
