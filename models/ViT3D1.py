import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights
from AAL_APE import AAL_APE
from AAL_APE.patch_utils import compute_patch_centers
from TokenTrimmer.TokenTrimmer import TokenTrimmer
import nibabel as nib
from REAM.REAM import REAM
#2 3都加
# 🔧 自动补齐体积到 patch_size 的倍数
def pad_to_divisible(x, patch_size):
    _, _, d, h, w = x.shape
    pd, ph, pw = patch_size
    dd = (pd - d % pd) if d % pd != 0 else 0
    dh = (ph - h % ph) if h % ph != 0 else 0
    dw = (pw - w % pw) if w % pw != 0 else 0
    # 注意：F.pad 的顺序是 (W前,W后,H前,H后,D前,D后)
    x = F.pad(x, (0, dw, 0, dh, 0, dd))
    return x

# ✅ 3D Patch Embedding 替代 ViT 中的 2D patch
class PatchEmbed3D(nn.Module):
    def __init__(self, in_channels=1, embed_dim=768, patch_size=(16,16,16)):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):  # x: [B,1,D,H,W]
        x = pad_to_divisible(x, self.patch_size)
        x = self.proj(x)                 # [B, embed_dim, D', H', W']
        x = x.flatten(2).transpose(1,2)  # [B, N, embed_dim]
        return x

# ✅ 整合到 ViT 模型
class ViT3D(nn.Module):
    def __init__(self, num_classes=4, patch_size=(16,16,16)):
        super().__init__()
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

        self.hidden_dim = self.vit.hidden_dim  # 保存方便用

        # 替换 patch embedding 为 3D patch
        self.patch_embed = PatchEmbed3D(
            in_channels=1,
            embed_dim=self.vit.hidden_dim,
            patch_size=patch_size
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        embed_dim = self.hidden_dim
        self.aal_pe = AAL_APE.AALPositionalEmbedding(
            aal_path="./AAL_APE/AAL2.nii.gz",
            embed_dim=embed_dim,
            region_max=116  # AAL 结构数量
        )

        self.pos_embed = None
        # 删除原始 ViT 的位置编码逻辑
        encoder = self.vit.encoder
        del self.vit.encoder

        self.encoder_blocks = nn.Sequential(*encoder.layers)
        self.encoder_norm = encoder.ln

        # 替换分类头
        self.heads = nn.Linear(self.vit.hidden_dim, num_classes)

        # 你可以把 M（输出 token 数）设成你想要的值，比如 128
        self.token_learner = TokenLearner(dim=self.hidden_dim, num_output_tokens=128)

        # REAM（模块3）
        self.ream = REAM(embed_dim=self.hidden_dim, num_regions=116)

        # def forward(self, x):  # x: [B,1,D,H,W]
    #     return self.vit(x)
    def forward(self, x, mri_affine, original_shape):
        B = x.shape[0]

        # 1) PatchEmbedding + cls_token
        x = self.patch_embed(x)  # [B, N, C]
        cls = self.cls_token.expand(B, -1, -1)  # [B,1,C]
        x = torch.cat([cls, x], dim=1)  # [B, N+1, C]

        # 2) AAL-APE 得到位置编码 + region_ids
        patch_centers = compute_patch_centers(
            img_shape=original_shape,
            patch_size=self.patch_embed.patch_size,
            padding=(11, 7, 3)
        )  # [N,3]
        patch_centers = patch_centers.unsqueeze(0).expand(B, -1, -1).to(x.device)
        region_pe, region_ids = self.aal_pe(patch_centers, mri_affine, return_ids=True)
        cls_pe = torch.zeros((B, 1, self.hidden_dim), device=x.device)
        region_pe = torch.cat([cls_pe, region_pe], dim=1)  # [B, N+1, C]
        x = x + region_pe  # [B, N+1, C]

        # 拆出 patch tokens（去掉 CLS）
        cls_token = x[:, :1, :]  # [B,1,C]
        patch_tokens = x[:, 1:, :]  # [B, N, C]

        # 3) 并行调用
        # 3a) 模块2：TokenLearner → 微观 M 个 token
        tl_out = self.token_learner(patch_tokens)  # [B, M, C]
        # 3b) 模块3：REAM → 宏观 N 个增强 patch
        ream_out = self.ream(patch_tokens, region_ids)  # [B, N, C]

        # 4) 拼接 CLS + tl_out + ream_out，再送编码器
        x = torch.cat([cls_token, tl_out, ream_out], dim=1)  # [B, 1+M+N, C]
        x = self.encoder_blocks(x)
        x = self.encoder_norm(x)
        logits = self.heads(x[:, 0])  # [B, num_classes]
        return logits
