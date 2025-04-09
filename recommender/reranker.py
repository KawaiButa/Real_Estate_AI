import torch.nn as nn
import torch
import numpy as np

class Reranker(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        return self.model(x)
def build_rerank_data(df, search_emb_cols, two_tower):
    u_input = df[["age", "min_price", "max_price", "lat", "lon"]].values.astype(np.float32)
    search_input = df[search_emb_cols].values.astype(np.float32)
    user_input = np.concatenate([u_input, search_input], axis=1)
    # For properties, use: price, rating, freshness, latitude, longitude, type_enc
    p_input = df[
        ["price", "rating", "freshness", "latitude_prop", "longitude_prop", "type_enc"]
    ].values.astype(np.float32)
    targets = df["score"].values.astype(np.float32)
    u_tensor = torch.tensor(user_input, dtype=torch.float32)
    p_tensor = torch.tensor(p_input, dtype=torch.float32)
    with torch.no_grad():
        u_emb = two_tower.user_tower(u_tensor).numpy()
        p_emb = two_tower.prop_tower(p_tensor).numpy()
    X = np.concatenate([u_emb, p_emb], axis=1)
    return X, targets
