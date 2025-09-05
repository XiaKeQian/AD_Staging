import torch
import torch.nn as nn

class TokenTrimmer(nn.Module):
    def __init__(self, dim, num_output_tokens):
        super().__init__()
        self.num_output_tokens = num_output_tokens
        self.attention_maps = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_output_tokens),
            nn.Softmax(dim=1)  # 对每个token维度归一化
        )

    def forward(self, x):  # x: [B, N, D]
        attn = self.attention_maps(x)       # [B, N, M]
        out = torch.einsum('bnd, bnm -> bmd', x, attn)  # [B, M, D]
        return out
