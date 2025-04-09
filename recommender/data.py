from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from dotenv import load_dotenv
load_dotenv()
search_model = SentenceTransformer("all-MiniLM-L6-v2")

class RecommenderDataset(Dataset):
    def __init__(self, user_feats, prop_feats, targets):
        self.user_feats = user_feats
        self.prop_feats = prop_feats
        self.targets = targets
    def __len__(self):
        return len(self.targets)
    def __getitem__(self, idx):
        return {"user": self.user_feats[idx],
                "property": self.prop_feats[idx],
                "target": self.targets[idx]}
def get_search_embedding(query):
    if pd.isnull(query) or query.strip() == "":
        return np.zeros(16)
    return search_model.encode(query)
