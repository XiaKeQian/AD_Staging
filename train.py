import torch
from torch.utils.data import DataLoader
from dataset import NPYDataset
from models.ViT3D import ViT3D
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load Dataset
train_set = NPYDataset("data/labels/index.pkl")
train_loader = DataLoader(train_set, batch_size=2, shuffle=True)

# 2. Init Model
model = ViT3D(num_classes=4).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 3. Train
for epoch in range(100):
    total_loss = 0
    correct = 0
    total = 0

    for batch in train_loader:
        images = batch['image'].to(device)         # [B,1,D,H,W]
        labels = batch['label'].to(device)         # [B]
        affines = batch['affine']  # List of [4,4] tensor
        shapes = batch['original_shape']  # List of (D, H, W)

        logits = []

        # 逐样本处理（因 affine 不可 batch）
        for i in range(images.shape[0]):
            img = images[i].unsqueeze(0).to(device)  # [1, 1, D, H, W]
            affine = affines[i].to(device)

            # 安全转换为元组
            shape = tuple(shapes[i].tolist())  # ← 这里明确转换成标准Python元组
            # print(f"shape[{i}] = {shapes[i]}, after conversion = {tuple(int(x) for x in shapes[i])}")

            out = model(img, affine, shape)  # [1, 4]
            logits.append(out)                  # [B, 4]

        logits = torch.cat(logits, dim=0)  # [B, 4]
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    print(f"Epoch {epoch+1}: Loss={total_loss:.3f}, Acc={correct/total:.3f}")
