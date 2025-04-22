import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, BertConfig, BertModel
from encoder import SearchActivityEncoder


class EnhancedUserTower(nn.Module):
    def __init__(
        self,
        input_dim=20,
        hidden_dim=64,
        out_dim=32,
        search_emb_dim=32,
        num_heads=4,
        num_layers=2,
        with_search=True,
    ):
        super().__init__()
        config = BertConfig(
            hidden_size=hidden_dim,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=hidden_dim * 4,
        )
        self.with_search = with_search
        self.user_input_dim = input_dim - search_emb_dim
        self.feature_proj = nn.Linear(
            self.user_input_dim if with_search else input_dim,
            hidden_dim,
        )
        self.transformer = BertModel(config)
        self.pooler = nn.Linear(hidden_dim, out_dim)
        self.search_fusion = nn.Sequential(
            nn.Linear(out_dim + search_emb_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, encoding):
        if self.with_search:
            feat = encoding[:, : self.user_input_dim]
            search_emb = encoding[:, self.user_input_dim :]
        else:
            feat = encoding
        x = self.feature_proj(feat)
        out = self.transformer(inputs_embeds=x.unsqueeze(1)).last_hidden_state.squeeze(1)
        user_emb = self.pooler(out)
        if self.with_search:
            user_emb = self.search_fusion(torch.cat([user_emb, search_emb], dim=-1))
        return F.normalize(user_emb, dim=1)


class TextTower(nn.Module):
    def __init__(self, model_name="huawei-noah/TinyBERT_General_6L_768D",
                 proj_hidden=256, out_dim=128):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        dim = self.encoder.config.hidden_size
        self.proj = nn.Sequential(
            nn.Linear(dim, proj_hidden), nn.ReLU(),
            nn.Linear(proj_hidden, out_dim)
        )

    def forward(self, input_ids, attention_mask):
        # CLS token pooling
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0]
        return self.proj(cls)


class StructuredTower(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        return self.mlp(x)


class MultiModalAttentionFusion(nn.Module):
    """
    Fuse text, structured, and review embeddings via self-attention.
    """
    def __init__(self, dim, fusion_dim, num_heads=4):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.proj = nn.Sequential(
            nn.Linear(dim, fusion_dim), nn.ReLU(),
            nn.Linear(fusion_dim, fusion_dim)
        )

    def forward(self, text_emb, struct_emb, review_emb):
        # stack modalities as sequence length=3
        x = torch.stack([text_emb, struct_emb, review_emb], dim=1)
        attn_out, _ = self.mha(x, x, x)
        pooled = attn_out.mean(dim=1)
        return self.proj(pooled)


class PropertyTower(nn.Module):
    def __init__(
        self,
        struct_dim,
        text_hidden_dim,
        fusion_dim,
        out_dim,
        num_views=3,
        dropout=0.2,
        noise_std=0.01,
        review_model="huawei-noah/TinyBERT_General_6L_768D",
        review_proj=128,
        review_out=64,
    ):
        super().__init__()
        # shared encoder for text & reviews
        self.base_text = AutoModel.from_pretrained(review_model)
        # separate projection heads
        self.text_proj = nn.Sequential(
            nn.Linear(self.base_text.config.hidden_size, text_hidden_dim), nn.ReLU(),
            nn.Linear(text_hidden_dim, fusion_dim)
        )
        self.review_proj = nn.Sequential(
            nn.Linear(self.base_text.config.hidden_size, review_proj), nn.ReLU(),
            nn.Linear(review_proj, fusion_dim)
        )
        self.structured_tower = StructuredTower(struct_dim, fusion_dim)
        self.fusion = MultiModalAttentionFusion(fusion_dim, fusion_dim)

        self.dropout = nn.Dropout(dropout)
        self.noise_std = noise_std
        self.views = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim//2), nn.ReLU(),
                nn.Linear(fusion_dim//2, out_dim)
            ) for _ in range(num_views)
        ])
        self.contrastive = nn.Linear(out_dim, out_dim)
        
    def _encode_reviews(self, review_seq):
        ids, mask = review_seq['input_ids'], review_seq['attention_mask']
        B, R, L = ids.size()
        
        if R == 0:
            # Return a zero tensor with expected output shape: (B, D)
            out_dim = self.review_proj[-1].out_features  # get the output dimension of the last layer in review_proj
            return torch.zeros(B, out_dim, device=ids.device)
        
        # Flatten to (B*R, L)
        flat_ids = ids.view(-1, L)
        flat_mask = mask.view(-1, L)

        # Forward through base text encoder
        out = self.base_text(input_ids=flat_ids, attention_mask=flat_mask).last_hidden_state[:, 0]  # shape: (B*R, D)

        # Project and reshape
        proj = self.review_proj(out).view(B, R, -1)

        # Average over reviews
        return proj.mean(dim=1)

    def forward(self, struct_feat, text_seq, review_seq=None):
        # text
        t = self.base_text(input_ids=text_seq['input_ids'], attention_mask=text_seq['attention_mask']).last_hidden_state[:,0]
        text_emb = self.text_proj(t)
        # struct
        s = self.structured_tower(struct_feat)
        # review
        if review_seq is not None:
            r = self._encode_reviews(review_seq)
        else:
            # zero vector fallback
            r = torch.zeros_like(text_emb)

        fused = self.fusion(text_emb, s, r)
        # multi-view
        vs = []
        for m in self.views:
            x = self.dropout(fused)
            x = x + torch.randn_like(x) * self.noise_std
            vs.append(m(x))
        avg = torch.stack(vs).mean(dim=0)
        return F.normalize(avg, dim=1), [F.normalize(self.contrastive(v), dim=1) for v in vs]


class TwoTowerRec(nn.Module):
    def __init__(
        self,
        search_args: dict,
        user_input_dim: int,
        prop_args: dict,
    ):
        super().__init__()
        self.search_enc = SearchActivityEncoder(**search_args)
        self.user_tower = EnhancedUserTower(input_dim=user_input_dim, with_search=False)
        self.prop_tower = PropertyTower(**prop_args)

    def forward(self, user_feat, prop_text, prop_struct, review_text=None, search=None):
        if search is not None:
            s_emb = self.search_enc(search)
            u_emb = self.user_tower(torch.cat([user_feat, s_emb], dim=-1))
        else:
            u_emb = self.user_tower(user_feat)
        p_emb, p_views = self.prop_tower(prop_text, prop_struct, review_text)
        return u_emb, p_emb, p_views

def cross_view_contrastive_loss(views, temp=0.07):
    loss = 0.0
    count = 0
    B = views[0].size(0)
    for i in range(len(views)):
        for j in range(i+1, len(views)):
            l = torch.matmul(views[i], views[j].T) / temp
            tgt = torch.arange(B, device=l.device)
            loss += (F.cross_entropy(l, tgt) + F.cross_entropy(l.T, tgt)) / 2
            count += 1
    return loss / count



class SoftContrastiveLoss(torch.nn.Module):
    def __init__(self, margin: float = 1.0, temp: float = 0.3,  lambda_ortho: float =0.1,):
        super().__init__()
        self.margin = margin
        self.temp = temp
        self.lambda_ortho = lambda_ortho
    def forward(self, u_emb, p_emb, p_views, t):
        # z_u, z_i: [batch, dim]; t: [batch] floats in [0,1]
        dist = F.pairwise_distance(u_emb, p_emb)  # [batch]
        pull = t * dist.pow(2)  # if t=1 → purely pull
        push = (1 - t) * F.relu(self.margin - dist).pow(2)
        # Multi-view contrastive loss
        view_loss = cross_view_contrastive_loss(p_views, self.temp)
        ortho_loss = torch.mean(torch.abs(torch.matmul(u_emb.T, p_emb)))
        loss = (pull + push).mean()
        total_loss = loss + view_loss * self.temp + ortho_loss * self.lambda_ortho
        print(
            f"Total loss: {total_loss} | MSE: {loss} | VIEW: {view_loss} | ORTHO: {ortho_loss}"
        )
        return loss
