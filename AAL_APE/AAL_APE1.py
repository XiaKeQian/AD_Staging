import torch
import torch.nn as nn
import nibabel as nib
import numpy as np
import os

class AALPositionalEmbedding(nn.Module):
    def __init__(self, aal_path, embed_dim=768, region_max=116, default_region_id=0):
        super().__init__()
        self.aal_path = aal_path
        self.embed_dim = embed_dim
        self.region_max = region_max
        self.default_region_id = default_region_id

        # 初始化嵌入表（region_id → vector）
        self.region_embed = nn.Embedding(region_max + 1, embed_dim)

        # 加载 AAL 图谱并提取 affine
        self.aal_img = nib.load(aal_path)
        self.aal_affine = self.aal_img.affine
        self.aal_data = self.aal_img.get_fdata()

    def forward(self, patch_centers_voxels, mri_affine):
        """
        输入：
            patch_centers_voxels: [B, N, 3]，每个 patch 中心在 MRI 图像中的 voxel 坐标
            mri_affine: [4, 4]，当前 MRI 图像的 affine 矩阵
        输出：
            region_embeddings: [B, N, embed_dim]，每个 patch 对应的位置向量
        """
        B, N, _ = patch_centers_voxels.shape
        device = patch_centers_voxels.device

        # Step 1: MRI voxel → MNI space
        ones = torch.ones((B, N, 1), device=device)
        voxel_homo = torch.cat([patch_centers_voxels.float(), ones], dim=-1)  # [B, N, 4]
        world_coords = torch.einsum('ij,bnj->bni', mri_affine, voxel_homo)  # [B, N, 4]

        # Step 2: MNI → AAL voxel
        inv_aal_affine = torch.tensor(np.linalg.inv(self.aal_affine), device=device, dtype=torch.float32)
        aal_voxel_coords = torch.einsum('ij,bnj->bni', inv_aal_affine, world_coords)  # [B, N, 4]
        aal_voxel_coords = aal_voxel_coords[..., :3].round().long()  # 取整以便索引

        # Step 3: 查 region ID（遍历坐标查 voxel 值）
        region_ids = torch.full((B, N), self.default_region_id, dtype=torch.long, device=device)

        D, H, W = self.aal_data.shape
        aal_tensor = torch.tensor(self.aal_data, device=device)

        # for b in range(B):
        #     for n in range(N):
        #         x, y, z = aal_voxel_coords[b, n]
        #         if 0 <= x < D and 0 <= y < H and 0 <= z < W:
        #             region_ids[b, n] = aal_tensor[x, y, z].long()
        #         else:
        #             region_ids[b, n] = self.default_region_id  # 越界设为默认脑外区域（如0）
        for b in range(B):
            for n in range(N):
                x, y, z = aal_voxel_coords[b, n]
                if 0 <= x < D and 0 <= y < H and 0 <= z < W:
                    region = aal_tensor[x, y, z].long().item()
                    if 0 <= region <= self.region_max:
                        region_ids[b, n] = region
                    else:
                        region_ids[b, n] = self.default_region_id
                else:
                    region_ids[b, n] = self.default_region_id

        # Step 4: 获取 region ID 对应的嵌入向量
        region_embeddings = self.region_embed(region_ids)  # [B, N, embed_dim]

        return region_embeddings

