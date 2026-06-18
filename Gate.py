import torch
import torch.nn as nn




class WeightedLearningGate(nn.Module):
    def __init__(self, hidden_dim, embed_dim, dropout: float = 0.2):
        super().__init__()
        assert hidden_dim == embed_dim, "建议 hidden_dim == embed_dim"

        input_dim = hidden_dim * 2
        self.gate_net = nn.Sequential(nn.Linear(input_dim, hidden_dim),)
        self.cand_net = nn.Sequential(nn.Linear(input_dim, hidden_dim),)

    def forward(self, knowledge_state, fused_features):
        gate_input = torch.cat([knowledge_state, fused_features], dim=-1)

        learning_raw = self.gate_net(gate_input)
        cand_raw = self.cand_net(gate_input)

        learning_gate = torch.sigmoid(learning_raw)
        candidate = torch.tanh(cand_raw)

        return learning_gate, candidate


class WeightedForgetGate(nn.Module):
    def __init__(self, hidden_dim, embed_dim, dropout: float = 0.2):
        super().__init__()
        assert hidden_dim == embed_dim

        input_dim = hidden_dim * 2

        self.gate_net = nn.Sequential(nn.Linear(input_dim, hidden_dim),)

    def forward(self, knowledge_state, fused_features):
        gate_input = torch.cat([knowledge_state, fused_features], dim=-1)

        forget_raw = self.gate_net(gate_input)
        forget_gate = torch.sigmoid(forget_raw)

        return forget_gate






class CJDLoss(nn.Module):
    def __init__(self, hidden_dim=128, lambd=0.0051):
        super().__init__()
        self.lambd = lambd

        self.projector_long = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim, bias=False)
        )

        self.projector_short = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim, bias=False)
        )

        self.bn = nn.BatchNorm1d(hidden_dim, affine=False)

    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, h_long, h_short):
        z_long = self.projector_long(h_long)
        z_short = self.projector_short(h_short)

        c = self.bn(z_long).T @ self.bn(z_short)
        c = c / h_long.size(0)

        on_diag = torch.diagonal(c).add(-1).pow(2).sum()
        off_diag = self.off_diagonal(c).pow(2).sum()

        return on_diag + self.lambd * off_diag


