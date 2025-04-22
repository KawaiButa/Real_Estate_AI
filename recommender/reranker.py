import torch
import torch.nn as nn


class Reranker(nn.Module):
    """
    An enhanced reranker model with additional layers and techniques for improved accuracy.

    Args:
        input_dim (int): Dimension of the concatenated user and property embeddings.
        hidden_dim (int): Dimension of the hidden layers.
        output_dim (int): Dimension of the output (default: 1 for scoring).
    """

    def __init__(self, input_dim=64, hidden_dim=128, output_dim=1):
        super(Reranker, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2), 
            nn.Linear(hidden_dim, hidden_dim), 
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim), 
        )

    def forward(self, x):
        """
        Forward pass to compute scores.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, output_dim].
        """
        return self.model(x)
