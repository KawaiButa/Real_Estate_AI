import torch
import torch.nn as nn

class UserTower(nn.Module):
    def __init__(self, input_dim=21, hidden_dim=64, out_dim=32):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, x):
        return self.model(x)
class PropertyTower(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, out_dim=32):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, x):
        return self.model(x)
class TwoTowerRec(nn.Module):
    def __init__(self):
        super().__init__()
        self.user_tower = UserTower()
        self.prop_tower = PropertyTower()
    def forward(self, user, prop):
        return self.user_tower(user), self.prop_tower(prop)
def cosine_loss(u, p, margin=0.5):
    cos_sim = nn.functional.cosine_similarity(u, p)
    loss = torch.clamp(margin - cos_sim, min=0).mean()
    return loss