import torch
import numpy as np
import faiss

def build_faiss_index(model, prop_df):
    prop_vec = prop_df[["price", "rating", "freshness", "latitude", "longitude", "type_enc"]].values.astype(np.float32)
    ids = prop_df["id"].values
    prop_tensor = torch.tensor(prop_vec, dtype=torch.float32)
    with torch.no_grad():
        emb = model.prop_tower(prop_tensor).numpy()
    faiss.normalize_L2(emb)
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)
    faiss.write_index(index, "property_faiss.index")
    np.save("property_id_map.npy", ids)
