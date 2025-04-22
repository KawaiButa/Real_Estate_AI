from collections import defaultdict
from sklearn.metrics import ndcg_score
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

# Custom modules
from data import clean_html_description
from tower import SoftContrastiveLoss, TwoTowerRec

# ------------------------- Config -------------------------
DATA_DIR = "/content"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")  # Expedia train file
MODEL_DIR = "checkpoints"
METRICS_FILE = os.path.join(MODEL_DIR, "validation_metrics.csv")

BATCH_SIZE = 128
NUM_EPOCHS = 10
TEST_SPLIT = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure checkpoint directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_6L_768D")

# Fields for Expedia two-tower
USER_FIELD_LIST = [
    "visitor_hist_starrating",
    "visitor_hist_adr_usd",
    "visitor_location_country_id",
    "visitor_location_region_id",
    "visitor_location_city_id",
    "date_time",
    "site_id",
    "posa_continent",
    "channel",
    "srch_ci",
    "srch_co",
    "srch_length_of_stay",
    "srch_booking_window",
    "srch_adults_count",
    "srch_children_count",
    "srch_room_count",
    "srch_saturday_night_bool",
    "srch_query_affinity_score",
    "orig_destination_distance",
    "random_bool",
    "srch_destination_id",
    "srch_destination_type_id",
]

PROPERTY_FIELD_LIST = [
    "prop_id",
    "prop_country_id",
    "prop_brand_bool",
    "prop_starrating",
    "prop_review_score",
    "prop_location_score1",
    "prop_location_score2",
    "prop_log_historical_price",
    "price_usd",
    "promotion_flag",
]

# ------------------------- Load and preprocess data -------------------------
data = pd.read_csv(TRAIN_CSV)

# Assign graded relevance for evaluation
# 5: booking, 1: click, 0: none
if 'booking_bool' in data.columns and 'click_bool' in data.columns:
    data['score'] = data['booking_bool'].fillna(0) * 5 + data['click_bool'].fillna(0) * 1
else:
    data['score'] = 1.0  # default

# ------------------------- Negative Sampling -------------------------

def generate_negative_samples(df, num_neg_per_user=5):
    sessions = df['srch_id'].dropna().unique()
    properties = df['prop_id'].dropna().unique()

    pos_dict = df.groupby('srch_id')['prop_id'].apply(set).to_dict()
    neg_rows = []

    for sid in tqdm(sessions, desc="Sampling negatives"):
        pos_props = pos_dict.get(sid, set())
        neg_cands = list(set(properties) - pos_props)
        sampled = np.random.choice(neg_cands, size=min(num_neg_per_user, len(neg_cands)), replace=False)

        sess_vals = df[df['srch_id'] == sid].iloc[0]
        for pid in sampled:
            prop_vals = df[df['prop_id'] == pid].iloc[0]
            new_row = prop_vals.copy()
            new_row['srch_id'] = sid
            new_row['score'] = 0.0
            for fld in USER_FIELD_LIST:
                new_row[fld] = sess_vals[fld]
            neg_rows.append(new_row)

    return pd.concat([df, pd.DataFrame(neg_rows)], ignore_index=True)

# Generate negatives
data = generate_negative_samples(data, num_neg_per_user=5)

# Fill missing
for col in USER_FIELD_LIST + PROPERTY_FIELD_LIST + ['score']:
    data[col] = data[col].fillna(0)

# Date parsing example
data['date_time'] = pd.to_datetime(data['date_time'], errors='coerce').fillna(pd.Timestamp(1970,1,1))
data['year'] = data['date_time'].dt.year
USER_FIELD_LIST = [fld for fld in USER_FIELD_LIST if fld != 'date_time'] + ['year']

# ------------------------- Dataset and Collate -------------------------
class ExpediaRecDataset(Dataset):
    def __init__(self, df):
        self.users = df[USER_FIELD_LIST].values.astype(np.float32)
        self.props = df[PROPERTY_FIELD_LIST].values.astype(np.float32)
        self.scores = df['score'].values.astype(np.float32)

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):
        return {
            'user': self.users[idx],
            'property': self.props[idx],
            'target': self.scores[idx]
        }

def collate_fn(batch):
    users = torch.from_numpy(np.stack([b['user'] for b in batch])).float()
    props = torch.from_numpy(np.stack([b['property'] for b in batch])).float()
    targets = torch.from_numpy(np.array([b['target'] for b in batch], dtype=np.float32)).float()
    return users, props, targets

# ------------------------- Split and Load -------------------------
dataset = ExpediaRecDataset(data)
num_test = int(len(dataset) * TEST_SPLIT)
num_train = len(dataset) - num_test
train_ds, val_ds = random_split(dataset, [num_train, num_test])
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# ------------------------- Model, optimizer, loss -------------------------
two_tower = TwoTowerRec(
    user_input_dim=len(USER_FIELD_LIST),
    property_input_dim=len(PROPERTY_FIELD_LIST),
).to(DEVICE)
optimizer = optim.AdamW(two_tower.parameters(), lr=1e-5)
loss_fn = SoftContrastiveLoss(margin=1.0, temp=0.5)

# Initialize metrics
pd.DataFrame(columns=['epoch','train_loss','ndcg@38','hit@38','map@38']).to_csv(METRICS_FILE, index=False)

# ------------------------- Validation -------------------------
def validate(model, val_loader, k=38):
    model.eval()
    user_scores = defaultdict(list)
    user_preds = defaultdict(list)

    with torch.no_grad():
        for users, props, targets in val_loader:
            users, props = users.to(DEVICE), props.to(DEVICE)
            u_emb, p_emb, preds = model(users, props)
            preds = preds.cpu().numpy()
            scores = targets.cpu().numpy()
            session_ids = users[:, 0].cpu().numpy()  # assuming srch_id as first user feature

            for sid, pred, actual in zip(session_ids, preds, scores):
                user_preds[sid].append(pred)
                user_scores[sid].append(actual)

    ndcg_list, hit_list, map_list = [], [], []
    for sid in user_preds:
        y_true = np.array(user_scores[sid])
        y_pred = np.array(user_preds[sid])
        if len(y_true) < 1 or np.all(y_true == 0):
            continue

        # NDCG@k with graded relevance
        ndcg_val = ndcg_score([y_true], [y_pred], k=k)
        ndcg_list.append(ndcg_val)

        # Hit@k: whether any positive relevance >0 in top-k
        topk_idx = np.argsort(y_pred)[::-1][:k]
        hit = int(np.any(y_true[topk_idx] > 0))
        hit_list.append(hit)

        # MAP@k
        ap = 0
        rel_count = 0
        for i, idx in enumerate(topk_idx, start=1):
            if y_true[idx] > 0:
                rel_count += 1
                ap += rel_count / i
        map_list.append(ap / min(k, np.sum(y_true > 0)))

    num_users = len(ndcg_list)
    return {
        'ndcg@38': np.mean(ndcg_list) if ndcg_list else 0.0,
        'hit@38': np.mean(hit_list) if hit_list else 0.0,
        'map@38': np.mean(map_list) if map_list else 0.0
    }

# ------------------------- Training Loop -------------------------
for epoch in range(1, NUM_EPOCHS+1):
    two_tower.train()
    total_loss = 0
    for users, props, targets in tqdm(train_loader, desc=f"Train Epoch {epoch}/{NUM_EPOCHS}"):
        optimizer.zero_grad()
        u_emb, p_emb, preds = two_tower(users.to(DEVICE), props.to(DEVICE))
        loss = loss_fn(u_emb, p_emb, preds, targets.to(DEVICE))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    metrics = validate(two_tower, val_loader)
    print(f"Epoch {epoch}: Loss={avg_loss:.4f}, NDCG@38={metrics['ndcg@38']:.4f}")

    # save metrics
    dfm = pd.read_csv(METRICS_FILE)
    dfm = pd.concat([dfm, pd.DataFrame([{**{'epoch':epoch, 'train_loss':avg_loss}, **metrics}])], ignore_index=True)
    dfm.to_csv(METRICS_FILE, index=False)

    torch.save(two_tower.state_dict(), os.path.join(MODEL_DIR, f"two_tower_epoch_{epoch}.pt"))

print("Training complete.")
