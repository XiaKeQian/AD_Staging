# File: AAL_APE/AAL_APE.py

import torch
import torch.nn as nn
import nibabel as nib
import numpy as np

class AALPositionalEmbedding(nn.Module):
    def __init__(self, aal_path, embed_dim=768, region_max=116, default_region_id=0):
        super().__init__()
        self.embed_dim = embed_dim
        self.region_max = region_max
        self.default_region_id = default_region_id

        # 初始化可学习的 region embedding 表
        self.region_embed = nn.Embedding(region_max + 1, embed_dim)

        # 加载 AAL 图谱
        self.aal_img = nib.load(aal_path)
        self.aal_affine = self.aal_img.affine
        self.aal_data = self.aal_img.get_fdata()

    def forward(self, patch_centers_voxels, mri_affine, return_ids=False):
        """
        输入：
            patch_centers_voxels: [B, N, 3]，每个 patch 在 MRI 空间的 voxel 中心坐标
            mri_affine:           [4, 4] 或 [B, 4, 4]，MRI 仿射矩阵
            return_ids:           是否返回 region_ids
        输出：
            region_embeddings: [B, N, C]
            (可选) region_ids: [B, N]
        """
        B, N, _ = patch_centers_voxels.shape
        device = patch_centers_voxels.device

        # 构造齐次坐标 [B, N, 4]
        ones = torch.ones((B, N, 1), device=device)
        voxel_homo = torch.cat([patch_centers_voxels.float(), ones], dim=-1)  # [B, N, 4]

        # 如果 mri_affine 是 [4,4]，则扩展到 [B,4,4]
        if mri_affine.ndim == 2:
            mri_affine = mri_affine.unsqueeze(0).expand(B, -1, -1)            # [B, 4, 4]

        # MRI voxel → 世界坐标 (MNI) （批量版本）
        # world_coords[b] = mri_affine[b] @ voxel_homo[b].T
        world_coords = torch.einsum('bij,bnj->bni', mri_affine, voxel_homo)  # [B, N, 4]

        # 世界坐标 → AAL voxel (单例 AAL_affine)
        inv_aal_affine = torch.tensor(
            np.linalg.inv(self.aal_affine),
            device=device,
            dtype=torch.float32
        )
        aal_voxel_coords = torch.einsum('ij,bnj->bni', inv_aal_affine, world_coords)  # [B, N, 4]
        aal_voxel_coords = aal_voxel_coords[..., :3].round().long()  # 取前三维整数

        # 遍历获取 region_ids
        D, H, W = self.aal_data.shape
        aal_tensor = torch.tensor(self.aal_data, device=device)
        region_ids = torch.full((B, N), self.default_region_id, dtype=torch.long, device=device)

        for b in range(B):
            for n in range(N):
                x, y, z = aal_voxel_coords[b, n]
                if 0 <= x < D and 0 <= y < H and 0 <= z < W:
                    region = int(aal_tensor[x, y, z].item())
                    if 0 <= region <= self.region_max:
                        region_ids[b, n] = region

        # 根据 region_ids 拿 embedding
        region_embeddings = self.region_embed(region_ids)  # [B, N, C]

        if return_ids:
            return region_embeddings, region_ids
        return region_embeddings
