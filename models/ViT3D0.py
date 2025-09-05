import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights
from AAL_APE import AAL_APE
from AAL_APE.patch_utils import compute_patch_centers
from TokenTrimmer.TokenTrimmer import TokenTrimmer
import nibabel as nib
from REAM.REAM import REAM
#原版存档
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
    def forward(self, x, mri_affine, original_shape):  # 新增两个参数  # x: [B, 1, D, H, W]
        x = self.patch_embed(x)  # 3D patch embedding → [B, N, C]
        B, N, C = x.shape

        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # [B, 1, C]
        x = torch.cat((cls_token, x), dim=1)  # 添加分类 token → [B, N+1, C]


        # # 🔧 动态生成 position embedding
        # if (self.pos_embed is None) or (self.pos_embed.size(1) != x.size(1)):
        #     self.pos_embed = nn.Parameter(
        #         torch.zeros(1, x.size(1), C, device=x.device)
        #     )
        #     nn.init.trunc_normal_(self.pos_embed, std=0.02)
        #
        # x = x + self.pos_embed[:, :x.size(1)]  # 🔧 防止 patch 数不同时报错


        # =============== 🔥 加入结构位置编码 ===============
        # ✅ Padding 已知：
        # 输入体积 shape = (197, 233, 189)
        # patch size = (16, 16, 16)
        # padding = (D:11, H:7, W:3)
        patch_centers = compute_patch_centers(
            img_shape=original_shape,
            patch_size=self.patch_embed.patch_size,
            padding=(11, 7, 3)
        )  # [N, 3] voxel 中心点

        patch_centers = patch_centers.unsqueeze(0).expand(B, -1, -1).to(x.device)  # [B, N, 3]

        region_pe = self.aal_pe(patch_centers, mri_affine)  # [B, N, C]
        # ✅ 为 cls_token 添加结构位置编码（全 0 向量）
        cls_pe = torch.zeros((B, 1, C), device=x.device)  # [B, 1, C]
        region_pe = torch.cat([cls_pe, region_pe], dim=1)  # [B, N+1, C]


        x = x + region_pe
        # # 插入模块2时这里被修改
        # # 拆出 cls_token 和 patch_tokens
        # cls_token = x[:, :1, :]  # [B,1,C]
        # patch_tokens = x[:, 1:, :]  # [B,N,C]
        #
        # # ① 用 TokenLearner 做 token 筛选／压缩
        # #    learned_tokens: [B, M, C]
        # learned_tokens = self.token_learner(patch_tokens)
        #
        # # ② 把 cls_token 拼回 learned_tokens
        # x = torch.cat([cls_token, learned_tokens], dim=1)  # [B, 1+M, C]
        #
        # # ③ 继续送入 Transformer Encoder
        x = self.encoder_blocks(x)
        x = self.encoder_norm(x)
        out = self.heads(x[:, 0])
        return out

        # ================================================

        # x = self.encoder_blocks(x)  # ✅ 自己调用 block
        # x = self.encoder_norm(x)
        # x = self.heads(x[:, 0])  # 取 [CLS] token
        # return x
