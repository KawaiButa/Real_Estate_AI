import torch
import torch.nn as nn

class SearchActivityEncoder(nn.Module):
    def __init__(
        self,
        text_dim,       # dim of precomputed text embeddings
        embed_dim,      # for categorical features
        num_numeric,    # number of numeric features (e.g. min/max price)
        hidden_dim,
        out_dim,
        num_types,
        num_cities,
        pooling='mean',  # 'mean' or 'max'
    ):
        super().__init__()
        self.type_embedding = nn.Embedding(num_types, embed_dim)
        self.city_embedding = nn.Embedding(num_cities, embed_dim)

        fusion_input_dim = text_dim + 2 * embed_dim + num_numeric

        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

        self.pooling = pooling

    def forward(self, emb):
        """
        Args:
            emb: [B, N, text_dim + 1 + 1 + num_numeric]        
        Returns:
            [B, out_dim] - pooled representation per user
        """

        text_dim = self.type_embedding.embedding_dim * 2
        text_emb = emb[:, :, :text_dim]  # [B, N, text_dim]
        type_id = emb[:, :, text_dim].long()  # [B, N]
        city_id = emb[:, :, text_dim + 1].long()  # [B, N]
        numeric_feats = emb[:, :, text_dim + 2:]  # [B, N, num_numeric]

        type_emb = self.type_embedding(type_id)  # [B, N, embed_dim]
        city_emb = self.city_embedding(city_id)  # [B, N, embed_dim]

        fused = torch.cat([text_emb, type_emb, city_emb, numeric_feats], dim=-1)  # [B, N, fusion_input_dim]
        fused = self.fusion_mlp(fused)  # [B, N, out_dim]
        pooled = fused.mean(dim=1)  # [B, out_dim]

        return pooled
