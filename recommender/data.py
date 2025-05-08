from datetime import datetime, timezone
import math
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from dotenv import load_dotenv

search_model = SentenceTransformer("all-MiniLM-L6-v2")


class RecommenderDataset(Dataset):
    def __init__(
        self,
        user_feats,
        prop_feats,
        search_feats,
        property_text_feats,
        targets,
        tokenizer,
    ):
        self.user_feats = user_feats
        self.prop_feats = prop_feats
        self.search_feats = search_feats
        self.property_text_feats = property_text_feats
        self.targets = targets
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return {
            "user": {"data": self.user_feats[idx], "searches": self.search_feats[idx]},
            "property": {
                "data": self.prop_feats[idx],
                "text": self.tokenizer(
                    self.property_text_feats[idx],
                    max_length=128,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                ),
            },
            "target": self.targets[idx],
        }


class PropertyEmbeddingDataset(Dataset):
    def __init__(self, property_feats: np.ndarray, property_texts: list[str]):
        """
        Args:
            property_feats: NumPy array of shape (N, D) with numerical features (already encoded).
            property_texts: List of cleaned property descriptions or titles.
        """
        assert len(property_feats) == len(
            property_texts
        ), "Features and texts must have the same length."
        self.property_feats = property_feats.astype(np.float32)
        self.property_texts = property_texts

    def __len__(self):
        return len(self.property_feats)

    def __getitem__(self, idx):
        return {
            "text": self.property_texts[idx],
            "data": self.property_feats[idx],
        }


def get_search_embedding(query):
    if pd.isnull(query) or query.strip() == "":
        return np.zeros(384)
    return search_model.encode(query)


def compute_score(row):
    decay = np.exp(-((datetime.now(tz=timezone.utc) - row["created_at"]).days) / 30)
    if row["action"] == "like":
        return 3 * decay
    elif row["action"] == "view":
        return 1 * decay
    elif row["action"] == "unlike":
        return -2 * decay
    return 0


def html_to_readable_text(html_text):
    soup = BeautifulSoup(html_text, "html.parser")
    for br in soup.find_all(["br", "p", "div"]):
        br.insert_before("\n")
    return soup.get_text(separator=" ", strip=True)


def clean_html_description(html_text):
    soup = BeautifulSoup(html_text, "html.parser")
    return soup.get_text(separator=" ", strip=True)


def encode_timestamp(timestamp):
    dt = datetime.fromtimestamp(timestamp)

    hour = dt.hour
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)

    weekday = dt.weekday()
    weekday_sin = np.sin(2 * np.pi * weekday / 7)
    weekday_cos = np.cos(2 * np.pi * weekday / 7)

    return np.array([hour_sin, hour_cos, weekday_sin, weekday_cos])


def sinusoidal_encode(row):
    # First normalize
    norm_lat = (row["lat"] + 90) / 180
    norm_lon = (row["lon"] + 180) / 360
    # Compute sine and cosine for lat and lon:
    sin_lat = np.sin(math.pi * norm_lat)
    cos_lat = np.cos(math.pi * norm_lat)
    sin_lon = np.sin(math.pi * norm_lon)
    cos_lon = np.cos(math.pi * norm_lon)
    return pd.Series(
        {"sin_lat": sin_lat, "cos_lat": cos_lat, "sin_lon": sin_lon, "cos_lon": cos_lon}
    )
def load_grouped_reviews(data_path: str, decay_rate: float):
    reviews = pd.read_csv(data_path, parse_dates=["date"])
    today = pd.to_datetime(datetime.now().date())
    reviews = reviews.dropna(subset=["comments"])
    reviews["comments"] = reviews["comments"].apply(clean_html_description)
    reviews["age_days"] = (today - reviews["date"]).dt.days.clip(lower=0)
    # compute decay weight
    reviews["weight"] = np.exp(-decay_rate * reviews["age_days"])
    grouped = (
    reviews.groupby("listing_id")
    .apply(
        lambda df: list(zip(df["comments"].astype(str).tolist(), df["weight"].tolist()))
    )
    .to_dict()
    )
    return grouped