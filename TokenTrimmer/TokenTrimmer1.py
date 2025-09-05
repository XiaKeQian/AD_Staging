import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenLearner(nn.Module):
    """
    TokenLearner 模块（双层 MLP 版本）

    输入:
        x: [B, N, D]，其中
            B = batch size
            N = token 数量（如 ViT 的 patch 数目）
            D = 每个 token 的特征维度（embedding dimension）

    参数:
        dim: int，输入 token 的特征维度 D
        num_output_tokens: int，要选出的输出 token 数量 K
        hidden_dim: int，中间隐藏层维度，默认为 dim // 2
    输出:
        out: [B, K, D]，K 个新生成的 token，每个 token 也是 D 维
    """

    def __init__(self, dim, num_output_tokens, hidden_dim=None):
        super().__init__()
        self.num_output_tokens = num_output_tokens

        # 如果不传 hidden_dim，就用 dim//2
        if hidden_dim is None:
            hidden_dim = dim // 2

        # Gate 网络：LayerNorm → Linear(dim -> hidden_dim) → GELU → Linear(hidden_dim -> num_output_tokens) → Softmax
        self.attention_maps = nn.Sequential(
            nn.LayerNorm(dim),  # 对最后一维 D 做 LayerNorm
            nn.Linear(dim, hidden_dim),  # 降维到 hidden_dim
            nn.GELU(),  # 非线性激活
            nn.Linear(hidden_dim, num_output_tokens),  # 映射到 K 个输出 token 的 raw scores
            nn.Softmax(dim=1)  # 对 token 轴 N 做 softmax（张量此时为 [B, N, K]，dim=1 即 N 轴）
        )

    def forward(self, x):
        """
        x: [B, N, D]
        返回:
            out: [B, K, D]
        """
        # 1. 计算 attention maps
        #    attn_raw: [B, N, K]
        attn_raw = self.attention_maps(x)

        # 2. 用 einsum 做加权和：out[b, k, d] = sum_{n=1..N} x[b, n, d] * attn_raw[b, n, k]
        #    这里的 attn_raw 维度 [B, N, K]，x 维度 [B, N, D]
        #    einsum 'bnd, bnk -> bkd' 先对 n 做加权和，得到 [B, K, D]
        out = torch.einsum('bnd, bnk -> bkd', x, attn_raw)

        return out
