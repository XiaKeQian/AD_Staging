import torch

def compute_patch_centers(img_shape, patch_size, padding):
    """
    计算 patch 中心点的 voxel 坐标

    参数:
        img_shape: 原始图像尺寸 (D, H, W)
        patch_size: patch 尺寸 (pD, pH, pW)
        padding: 每个维度的 padding 值 (padD, padH, padW)

    返回:
        Tensor [N, 3]，每个 patch 的中心点 voxel 坐标（int）
    """

    D, H, W = img_shape
    pD, pH, pW = patch_size
    padD, padH, padW = padding

    # 计算 padding 后的 shape
    padded_D = D + 2 * padD
    padded_H = H + 2 * padH
    padded_W = W + 2 * padW

    # 计算每个维度的 patch 数量
    nD = padded_D // pD
    nH = padded_H // pH
    nW = padded_W // pW

    centers = []

    for d in range(nD):
        for h in range(nH):
            for w in range(nW):
                cx = d * pD + pD // 2 - padD
                cy = h * pH + pH // 2 - padH
                cz = w * pW + pW // 2 - padW
                centers.append((cx, cy, cz))

    centers = torch.tensor(centers, dtype=torch.long)  # [N, 3]
    return centers
