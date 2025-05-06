import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
import pickle
import os

# 🔧 从 transform.mat 文件读取仿射矩阵
def read_fsl_affine(mat_path):
    with open(mat_path, 'r') as f:
        lines = f.readlines()
    matrix = []
    for line in lines:
        row = []
        for x in line.strip().split():
            try:
                row.append(float(x))
            except ValueError:
                row.append(float.fromhex(x))  # 支持 hex 格式 float
        matrix.append(row)
    return np.array(matrix)


class NPYDataset(Dataset):
    def __init__(self, index_file, transform=None):
        with open(index_file, 'rb') as f:
            self.index_dict = pickle.load(f)  # {PTID: (full_path, label)}

        # 转换成 [(path, label)] 列表方便索引
        self.samples = list(self.index_dict.values())
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npy_path, label = self.samples[idx]
        data = np.load(npy_path).astype(np.float32)  # [D,H,W]

        original_shape = torch.tensor(data.shape[-3:], dtype=torch.long)  # ← 改成tensor类型，安全返回

        mat_path = npy_path.replace("aligned.npy", "transform.mat")
        affine = read_fsl_affine(mat_path)


        if self.transform:
            data = self.transform(data)

        # 确保有 channel 维度：[1,D,H,W]
        if data.ndim == 3:
            data = np.expand_dims(data, axis=0)

        return {
            "image": torch.tensor(data),  # [1, D, H, W]
            "label": torch.tensor(label).long(),
            "affine": torch.tensor(affine).float(),  # 转为 tensor
            "original_shape": original_shape   # ✅ 只要 D, H, W
        }
