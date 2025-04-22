# -------------------- Inference: Retrieval + Reranking --------------------
import os
import faiss
import numpy as np
import pandas as pd
import torch

from reranker import Reranker
from tower import TwoTowerRec
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
property_field_list = ["price", "average_rating", "lat", "lon", "type_enc"]
num_types = 6
num_cities = 20
user_field_list = ["lat", "lon"]

two_tower = TwoTowerRec(
    search_text_dim=384,
    search_embed_dim=1,
    search_num_numeric=4,
    search_out_dim=32,
    search_num_cities=num_cities,
    search_num_types=num_types,
    user_input_dim=32 + len(user_field_list),
    property_input_dim=len(property_field_list),
)
two_tower_state_dict = torch.load("two_tower_rec.pth",map_location=torch.device("cpu"))
two_tower.load_state_dict(two_tower_state_dict)
reranker = Reranker()
reranker_state_dict = torch.load("reranker.pth", map_location=torch.device("cpu"))
reranker.load_state_dict(reranker_state_dict)
def retrieve_candidates(user_vector, top_k=20):
    user_tensor = torch.tensor(user_vector, dtype=torch.float32)
    with torch.no_grad():
        u_emb = two_tower.user_tower(user_tensor).numpy()
    faiss.normalize_L2(u_emb)
    index = faiss.read_index("property_faiss.index")
    ids = np.load("property_id_map.npy", allow_pickle=True)
    dists, idxs = index.search(u_emb, top_k)
    return ids[idxs[0]], u_emb
def load_property_df() -> pd.DataFrame:
    return pd.read_csv('property_features.csv')
property_df = load_property_df()
def recommend(
    two_tower, reranker, user_raw_vector, property_type_filter=None, boost=0.1
):
    user_raw = np.array([user_raw_vector], dtype=np.float32)
    candidate_ids, u_emb = retrieve_candidates(user_raw, top_k=20)
    print(candidate_ids)
    results = []
    for pid in candidate_ids:
        prop = property_df[property_df["id"] == str(pid)]
        p_vec = prop[property_field_list].values.astype(np.float32)
        p_tensor = torch.tensor(p_vec, dtype=torch.float32)
        with torch.no_grad():
            p_emb = two_tower.prop_tower(p_tensor).numpy()
        if len(p_emb) == 0: continue
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


example_user_features = np.array([0.0, 70000000, 17.444326112, 108.718540774] + [0] * 16, dtype=np.float32)
recommended_properties = recommend(
    two_tower, reranker, example_user_features, property_type_filter="Apartment"
)