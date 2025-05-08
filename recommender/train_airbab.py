import os
import math
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import torch.nn.functional as F
from datetime import datetime
from torch.utils.data import DataLoader, Dataset, random_split
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoTokenizer

from sklearn.metrics import ndcg_score
from data import clean_html_description
from tower import SoftContrastiveLoss, TwoTowerRec

# ------------------------- Config -------------------------
DATA_DIR = "/content"
LISTING_CSV = os.path.join(DATA_DIR, "listings.csv")
REVIEW_CSV = os.path.join(
    DATA_DIR, "reviews.csv"
)  # reviewer_id,id,date,reviewer_name,comments
DATA_CSV = os.path.join(DATA_DIR, "data.csv")
MODEL_DIR = "checkpoints"
METRICS_FILE = os.path.join(MODEL_DIR, "validation_metrics.csv")

BATCH_SIZE = 32
NUM_EPOCHS = 10
TEST_SPLIT = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyperparam for review decay: weight = exp(-decay_rate * age_days)
DECAY_RATE = 0.001
MAX_REVIEWS = 7  # limit per property for batching
MAX_LEN = 128

os.makedirs(MODEL_DIR, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_6L_768D")
# Fields
USER_FIELD_LIST = ["reviewer_id"]
PROPERTY_FIELD_LIST = [
    "latitude",
    "longitude",
    "accommodates",
    "bathrooms",
    "bedrooms",
    "price",
    "property_type",
]

# ----------------------- Load and merge reviews -----------------------
if __name__ == "__main__":
    reviews = pd.read_csv(REVIEW_CSV, parse_dates=["date"])  # assume 'date' column
    # compute age in days from today
    today = pd.to_datetime(datetime.now().date())
    reviews = reviews.dropna(subset=["comments"])
    reviews["comments"] = reviews["comments"].apply(clean_html_description)
    reviews["age_days"] = (today - reviews["date"]).dt.days.clip(lower=0)
    # compute decay weight
    reviews["weight"] = np.exp(-DECAY_RATE * reviews["age_days"])

    # group reviews by listing_id
    grouped = (
        reviews.groupby("listing_id")
        .apply(
            lambda df: list(
                zip(df["comments"].astype(str).tolist(), df["weight"].tolist())
            )
        )
        .to_dict()
    )


# ------------------------- Dataset -------------------------
class AirbnbRecDataset(Dataset):
    def __init__(self, df, grouped_reviews):
        # base data
        self.prop_ids = df["id_x"].values
        self.user_ids = df["id_y"].values
        self.users = df[USER_FIELD_LIST].values.astype(np.float32)
        self.props = df[PROPERTY_FIELD_LIST].values.astype(np.float32)
        self.texts = df["description"].values
        self.scores = df["score"].values.astype(np.float32)
        self.grouped_reviews = grouped_reviews

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):
        pid = int(self.prop_ids[idx])
        uid = int(self.user_ids[idx])
        reviews_list = self.grouped_reviews.get(pid, [])
        reviews_list = sorted(reviews_list, key=lambda x: -x[1])[:MAX_REVIEWS]

        texts = [x[0] for x in reviews_list]
        weights = [x[1] for x in reviews_list]
        R = len(texts)
        # tokenize all texts
        encodings = (
            tokenizer(
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
        return {
            "user_id": uid,
            "property_id": pid,
            "user": self.users[idx],
            "property": self.props[idx],
            "text": tokenizer(
                self.texts[idx],
                max_length=MAX_LEN,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ),
            "review_seq": encodings,
            "target": self.scores[idx],
        }


# collate
def collate_fn(batch):
    user_id_np = np.stack([b["user_id"] for b in batch], axis=0)
    user_ids = torch.from_numpy(user_id_np).float()
    prop_id_np = np.stack([b["property_id"] for b in batch], axis=0)
    prop_ids = torch.from_numpy(prop_id_np).float()
    user_np = np.stack([b["user"] for b in batch], axis=0)
    users = torch.from_numpy(user_np).float()
    prop_np = np.stack([b["property"] for b in batch], axis=0)
    props = torch.from_numpy(prop_np).float()
    texts = {
        k: torch.cat([b["text"][k] for b in batch], dim=0) for k in batch[0]["text"]
    }

    # pad reviews R dimension
    maxR = max(b["review_seq"]["input_ids"].shape[0] for b in batch)
    input_ids = []
    attention_mask = []
    for b in batch:
        r_ids = b["review_seq"]["input_ids"]
        r_mask = b["review_seq"]["attention_mask"]
        # pad to maxR
        pad_len = maxR - r_ids.shape[0]
        if pad_len > 0:
            r_ids = torch.cat(
                [r_ids, torch.zeros((pad_len, r_ids.size(1)), dtype=torch.long)], dim=0
            )
            r_mask = torch.cat(
                [r_mask, torch.zeros((pad_len, r_mask.size(1)), dtype=torch.long)],
                dim=0,
            )
        input_ids.append(r_ids.unsqueeze(0))
        attention_mask.append(r_mask.unsqueeze(0))
    review_seq = {
        "input_ids": torch.cat(input_ids, dim=0),  # [B, R, L]
        "attention_mask": torch.cat(attention_mask, dim=0),
    }

    targets = torch.tensor([b["target"] for b in batch], dtype=torch.float)
    return users, props, texts, review_seq, targets, user_ids, prop_ids


def generate_negative_samples(df, num_neg_per_user=7, score=0.69):
    users = df["reviewer_id"].dropna().unique()
    properties = df["id_x"].dropna().unique()  # Assuming 'id' is property_id

    user_pos_dict = df.groupby("reviewer_id")["id_x"].apply(set).to_dict()
    negative_rows = []

    for user in tqdm(users, desc="Sampling negatives"):
        pos_props = user_pos_dict[user]
        neg_candidates = list(set(properties) - pos_props)
        sampled_negs = np.random.choice(
            neg_candidates,
            size=min(num_neg_per_user, len(neg_candidates)),
            replace=False,
        )

        user_rows = df[df["reviewer_id"] == user].iloc[
            0
        ]  # grab user-level info (reuse)
        for pid in sampled_negs:
            prop_row = df[df["id_x"] == pid].iloc[0]  # grab property-level info
            new_row = prop_row.copy()
            new_row["reviewer_id"] = user
            new_row["score"] = score
            negative_rows.append(new_row)

    neg_df = pd.DataFrame(negative_rows)
    return pd.concat([df, neg_df], ignore_index=True)


from collections import defaultdict
import numpy as np
import torch
from sklearn.metrics import ndcg_score


def validate(model, val_loader, k=10, threshold=0.0):
    model.eval()
    session_scores = defaultdict(list)
    session_preds = defaultdict(list)

    with torch.no_grad():
        for users, props, texts, review_seq, targets, user_ids, prop_ids in val_loader:
            users, props = users.to(DEVICE), props.to(DEVICE)
            texts = {kk: vv.to(DEVICE) for kk, vv in texts.items()}
            review_seq = {kk: vv.to(DEVICE) for kk, vv in review_seq.items()}

            # use the passed model, not a global
            u_emb, p_emb, _ = model(users, props, texts, review_seq)
            dists = (F.pairwise_distance(u_emb, p_emb)).cpu().numpy()
            sigma = dists.mean()
            preds = np.exp(-(dists**2) / (2 * sigma**2))
            uids = users[:, 0].cpu().numpy()
            actual = targets.cpu().numpy()

            for uid, p, a in zip(uids, preds, actual):
                session_preds[uid].append(p)
                session_scores[uid].append(a)

    # accumulate per-session metrics
    ndcg_list, hit_list, map_list = [], [], []
    mrr_list, prec_list, rec_list, f1_list = [], [], [], []

    for uid in session_preds:
        y_true = np.array(session_scores[uid])
        y_pred = np.array(session_preds[uid])

        # skip sessions with no positive graded relevance or too few items
        if np.sum(y_true) <= 0 or len(y_true) < 2:
            continue

        # top‑k predicted indices
        idx = np.argsort(y_pred)[::-1][:k]

        # NDCG@k
        ndcg_list.append(ndcg_score([y_true], [y_pred], k=k))

        # binary relevance
        binary = (y_true >= threshold).astype(int)
        if binary.sum() == 0:
            continue
        rel = binary[idx]

        # Hit@k
        hit_list.append(int(rel.sum() > 0))

        # AP@k (for MAP)
        ap, hits = 0.0, 0
        for rank, i in enumerate(idx, start=1):
            if binary[i]:
                hits += 1
                ap += hits / rank
        map_list.append(ap / min(k, binary.sum()))

        # MRR@k
        rel_ranks = np.where(rel > 0)[0]
        mrr_list.append(1.0 / (rel_ranks[0] + 1) if rel_ranks.size > 0 else 0.0)

        # Precision@k and Recall@k
        prec = rel.sum() / k
        rec = rel.sum() / binary.sum()
        prec_list.append(prec)
        rec_list.append(rec)

        # F1@k
        if prec + rec > 0:
            f1_list.append(2 * prec * rec / (prec + rec))
        else:
            f1_list.append(0.0)

    n = len(ndcg_list)
    return {
        f"ndcg@{k}": np.mean(ndcg_list) if n else 0.0,
        f"hit@{k}": np.mean(hit_list) if n else 0.0,
        f"map@{k}": np.mean(map_list) if n else 0.0,
        f"mrr@{k}": np.mean(mrr_list) if n else 0.0,
        f"precision@{k}": np.mean(prec_list) if n else 0.0,
        f"recall@{k}": np.mean(rec_list) if n else 0.0,
        f"f1@{k}": np.mean(f1_list) if n else 0.0,
    }


# ------------------------- Load main data -------------------------
if __name__ == "__main__":
    data = pd.read_csv(DATA_CSV)
    # data = generate_negative_samples(data, num_neg_per_user=, score=0.0)
    # preprocessing as before
    property_type_list = data["property_type"].unique().tolist()
    data["property_type"] = data["property_type"].apply(
        lambda x: property_type_list.index(x)
    )

    data["price"] = data["price"].str.replace("[\$,]", "", regex=True).astype(float)
    # negative sampling omitted for brevity...
    # fill na
    for col in ["description", "reviewer_id", "score"]:
        data[col] = data[col].fillna(0 if col != "description" else "")
    data.fillna(0, inplace=True)

    # split and dataloaders
    dataset = AirbnbRecDataset(data, grouped)
    num_test = int(len(dataset) * TEST_SPLIT)
    train_ds, test_ds = random_split(dataset, [len(dataset) - num_test, num_test])
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    # ------------------------- Model & Training -------------------------
    search_args = {
        "text_dim": 384,
        "embed_dim": 1,
        "num_numeric": 0,
        "hidden_dim": 64,
        "out_dim": 32,
        "num_types": len(property_type_list),
        "num_cities": 0,
    }

    prop_args = {
        "struct_dim": len(PROPERTY_FIELD_LIST),
        "text_hidden_dim": 64,
        "fusion_dim": 256,
        "out_dim": 32,
    }

    two_tower = TwoTowerRec(
        search_args=search_args,
        user_input_dim=1,
        prop_args=prop_args,
    ).to(DEVICE)

    optimizer = optim.AdamW(two_tower.parameters(), lr=1e-4, weight_decay=1e-2)
    loss_fn = SoftContrastiveLoss(margin=1.0, temp=0.5, lambda_ortho=0.8)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

    # metrics file
    pd.DataFrame(
        columns=[
            "epoch",
            "train_loss",
            "ndcg@10",
            "hit@10",
            "map@10",
            "mrr@10",
            "precision@10",
            "recall@10",
        ]
    ).to_csv(METRICS_FILE, index=False)

    # training loop
    for epoch in range(1, NUM_EPOCHS + 1):
        two_tower.train()
        total_loss = 0.0
        index = 1
        current_lr = scheduler.get_last_lr()[0]
        epoch_bar = tqdm(train_loader, desc=f"Epoch {epoch}, LR={current_lr:.2e}")
        for users, props, texts, review_seq, targets, user_ids, prop_ids in epoch_bar:
            optimizer.zero_grad()
            users, props, user_ids, prop_ids = (
                users.to(DEVICE),
                props.to(DEVICE),
                user_ids.to(DEVICE),
                prop_ids.to(DEVICE),
            )
            texts = {k: v.to(DEVICE) for k, v in texts.items()}
            review_seq = {k: v.to(DEVICE) for k, v in review_seq.items()}
            targets = targets.to(DEVICE)
            u_emb, p_emb, p_views = two_tower(users, props, texts, review_seq)
            loss = loss_fn(u_emb, p_emb, p_views, targets, user_ids, prop_ids)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            index += 1
            epoch_bar.set_description(
                f"Epoch {epoch}, Loss={total_loss / index}, LR={current_lr:.2e}"
            )
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        # validation omitted for brevity; call validate same as before
        print(f"Epoch {epoch}: train_loss={avg_loss:.4f}")
        # Validation
        metrics = validate(two_tower, val_loader, k=5)

        print(f"Epoch {epoch}: Loss={avg_loss:.4f}, LR={current_lr:.2e}")
        for key, value in metrics.items():
            print(f"{key}: {value}")
        # Save metrics
        dfm = pd.read_csv(METRICS_FILE)
        new_row = {**{"epoch": epoch, "train_loss": avg_loss}, **metrics}
        dfm = pd.concat([dfm, pd.DataFrame([new_row])], ignore_index=True)
        dfm.to_csv(METRICS_FILE, index=False)  # save checkpoint
        torch.save(
            two_tower.state_dict(),
            os.path.join(MODEL_DIR, f"two_tower_epoch_{epoch}.pt"),
        )
    print("Training complete.")
