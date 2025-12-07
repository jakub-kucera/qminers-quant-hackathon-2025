import numpy as np
import torch
import torch.nn as nn
import base64
import io

TEAM_EMBED_LEN = 15
FEATURE_LEN = TEAM_EMBED_LEN + 12

WGHT="""
Someverylongstrajdnknfakjnfdsaln
"""

class MatchPredictor(nn.Module):
    def __init__(self, input_dim=FEATURE_LEN, hidden1=64, hidden2=32, output_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden2, output_dim),
        )

    def forward(self, x):
        # convert the bookmaker probs
        market_logits = x[:, -3:]
        return self.net(x) + market_logits

if __name__ == "__main__":
    # load the np arrays
    mean = np.load("scaler_mean.npy")
    var = np.load("scaler_var.npy")
    model=MatchPredictor()
    model.load_state_dict(torch.load("model.json"))

    print(WGHT)

    #buffer = io.BytesIO()
    #torch.save(model.state_dict(), buffer)
    #encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

    #print(encoded)
