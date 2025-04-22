import torch
import numpy as np
import faiss

from data import PropertyEmbeddingDataset

property_field_list = ["price", "average_rating", "lat", "lon", "type_enc"]

from torch.utils.data import DataLoader
from tqdm import tqdm


def build_faiss_index(
    model,
    property_df,
    batch_size=128,
    index_path="property_faiss.index",
    id_map_path="property_id_map.npy",
):
    """
    Builds a FAISS index for property embeddings using batched processing.

    Args:
        model: The model with a prop_tower(texts, features) method.
        dataset (RecommenderDataset): The dataset containing property features and text.
        property_ids (np.ndarray): An array of property IDs aligned with the dataset.
        batch_size (int): Batch size for processing.
        index_path (str): Path to save the FAISS index.
        id_map_path (str): Path to save the property ID map.
    """
    model.eval()
    dim = None
    all_embs = []
    dataset = PropertyEmbeddingDataset(
        property_df[property_field_list].values.astype(np.float32),
        property_df["text"].values,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for batch in tqdm(dataloader, desc="Building FAISS index"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_texts = batch["text"]
        batch_feats = batch["data"].to(device)

        with torch.no_grad():
            emb, _ = model.prop_tower(batch_texts, batch_feats)
            emb = emb.cpu().numpy()
            faiss.normalize_L2(emb)
            all_embs.append(emb)

    # Stack all embeddings
    all_embs = np.vstack(all_embs)

    # Create FAISS index
    dim = all_embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(all_embs)

    # Save the index and property ID mapping
    faiss.write_index(index, index_path)
    np.save(id_map_path, len(property_df["id"]))
