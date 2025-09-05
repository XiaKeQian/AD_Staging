# TokenLearner.py

import torch
import torch.nn as nn


class TokenTrimmer(nn.Module):
    """
    TokenLearner 模块（双层 MLP 版本 + AAL-PE 相加融合）

    输入:
        x: [B, N, D]，patch token 序列
        aal_pe: [B, N, D]，对应的 AAL-APE 结构位置编码

    参数:
        dim: int，输入 token 的特征维度 D
        num_output_tokens: int，要选出的输出 token 数量 K
        hidden_dim: int，中间隐藏层维度，默认为 dim // 2

    输出:
        out: [B, K, D]，K 个新生成的 token，每个 token 还是 D 维
    """

    def __init__(self, dim, num_output_tokens, hidden_dim=None):
        super().__init__()
        self.num_output_tokens = num_output_tokens

        # 如果不传 hidden_dim，就用 dim // 2
        if hidden_dim is None:
            hidden_dim = dim // 2

        # attention_maps: LayerNorm(dim) → Linear(dim→hidden_dim) → GELU → Linear(hidden_dim→K) → Softmax(dim=1)
        # 注意 Softmax 后得到的是 [B, N, K]，在 token 维度（dim=1）做归一化
        self.attention_maps = nn.Sequential(
            nn.LayerNorm(dim),                 # 对最后一维 D 做 LayerNorm
            nn.Linear(dim, hidden_dim),        # 降维到 hidden_dim
            nn.GELU(),                         # 非线性激活
            nn.Linear(hidden_dim, num_output_tokens),  # 映射到 K 个 raw scores
            nn.Softmax(dim=1)                  # 对 token 维度 N 做 Softmax
        )

    def forward(self, x, aal_pe):
        """
        x:     [B, N, D]
        aal_pe:[B, N, D]

        返回:
            out: [B, K, D]
        """

        # 1. 先把 AAL-PE 和输入 token 相加，得到融合特征 x_fuse
        #    这样 Softmax 得到的注意力权重就同时考虑了“原始 token”与“AAL 结构先验”。
        x_fuse = x + aal_pe   # [B, N, D]

        # 2. 计算 attention maps
        #    attn_raw: [B, N, K]
        attn_raw = self.attention_maps(x_fuse)

        # 3. 使用加权求和把 N 个 token 压缩到 K 个 token
        #    这里用的是“原始 token x”去加权，以保证输出 token 是纯视觉特征（加权时才考虑 aal 信息）
        #    einsum 公式：out[b, k, d] = sum_{n=1..N} x[b, n, d] * attn_raw[b, n, k]
        out = torch.einsum('bnd,bnk->bkd', x, attn_raw)

        # 返回形状 [B, K, D]
        return out
