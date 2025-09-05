import torch
import torch.nn as nn
import torch.nn.functional as F

class REAM(nn.Module):
    def __init__(self, embed_dim=768, num_regions=116):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_regions = num_regions

        # 可学习的 Region Tokens: [1, N, C]
        self.region_tokens = nn.Parameter(torch.randn(1, num_regions, embed_dim))

        # 第一次 Cross-Attention: region ← patch
        self.q_proj_region = nn.Linear(embed_dim, embed_dim)
        self.k_proj_patch  = nn.Linear(embed_dim, embed_dim)
        self.v_proj_patch  = nn.Linear(embed_dim, embed_dim)

        # 第二次 Cross-Attention: patch ← region
        self.q_proj_patch  = nn.Linear(embed_dim, embed_dim)
        self.k_proj_region = nn.Linear(embed_dim, embed_dim)
        self.v_proj_region = nn.Linear(embed_dim, embed_dim)

        # 输出融合层
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x_patch, region_ids):
        """
        x_patch: [B, P, C]      Patch tokens（含AAL位置编码）
        region_ids: [B, P]      每个 token 所属脑区编号（int ∈ [0, N-1]）
        return:     [B, P, C]   融合脑区语义后的 Patch tokens
        """
        B, P, C = x_patch.shape
        N = self.num_regions

        # 1. 扩展 region tokens
        region_tokens = self.region_tokens.expand(B, -1, -1)  # [B, N, C]

        # 2. region ← patch（第一次 Cross-Attention）
        Q = self.q_proj_region(region_tokens)        # [B, N, C]
        K = self.k_proj_patch(x_patch)               # [B, P, C]
        V = self.v_proj_patch(x_patch)               # [B, P, C]

        attn_scores1 = torch.matmul(Q, K.transpose(1, 2)) / (C ** 0.5)  # [B, N, P]

        # region-to-patch mask: [B, N, P]
        region_ids_exp = region_ids.unsqueeze(1).expand(B, N, P)        # [B, N, P]
        region_range = torch.arange(N, device=region_ids.device).view(1, N, 1)
        mask1 = (region_ids_exp == region_range).float()
        attn_scores1 = attn_scores1.masked_fill(mask1 == 0, float('-inf'))

        attn_weights1 = F.softmax(attn_scores1, dim=-1)  # [B, N, P]
        region_out = torch.matmul(attn_weights1, V)      # [B, N, C]

        # 3. patch ← region（第二次 Cross-Attention）
        Q2 = self.q_proj_patch(x_patch)             # [B, P, C]
        K2 = self.k_proj_region(region_out)         # [B, N, C]
        V2 = self.v_proj_region(region_out)         # [B, N, C]

        attn_scores2 = torch.matmul(Q2, K2.transpose(1, 2)) / (C ** 0.5)  # [B, P, N]

        # patch-to-region mask: [B, P, N]
        region_ids_exp2 = region_ids.unsqueeze(-1)                       # [B, P, 1]
        region_range2 = torch.arange(N, device=region_ids.device).view(1, 1, N)
        mask2 = (region_ids_exp2 == region_range2).float()
        attn_scores2 = attn_scores2.masked_fill(mask2 == 0, float('-inf'))

        attn_weights2 = F.softmax(attn_scores2, dim=-1)  # [B, P, N]
        patch_out = torch.matmul(attn_weights2, V2)      # [B, P, C]

        # 4. 融合输出（残差 + 线性）
        x_out = self.output_proj(x_patch + patch_out)    # [B, P, C]

        return x_out
