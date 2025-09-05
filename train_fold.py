import argparse
import torch
from torch.utils.data import DataLoader
from dataset import NPYDataset
from models.ViT3D import ViT3D
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 参数输入：fold 编号
parser = argparse.ArgumentParser()
parser.add_argument('--fold', type=int, default=0)
args = parser.parse_args()

# ✅ 路径加载
train_index = f"data/labels/kfold/train_fold{args.fold}.pkl"
val_index   = f"data/labels/kfold/val_fold{args.fold}.pkl"

train_set = NPYDataset(train_index)
val_set   = NPYDataset(val_index)

train_loader = DataLoader(train_set, batch_size=2, shuffle=True)
val_loader   = DataLoader(val_set, batch_size=2, shuffle=False)

# ✅ 初始化模型
model = ViT3D(num_classes=4).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ✅ 训练 + 验证循环
for epoch in range(2000):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in train_loader:
        img = batch['image'].to(device)  # [B, 1, D, H, W]
        label = batch['label'].to(device)  # [B]
        affine = batch['affine']  # List of [4,4] tensor
        shape = batch['original_shape']  # List of (D, H, W)

        logits = []

        for i in range(img.shape[0]):
            img_i = img[i].unsqueeze(0)  # [1, 1, D, H, W]
            affine_i = affine[i].to(device)
            shape_i = tuple(shape[i].tolist())  # 💡 安全转换
            out_i = model(img_i, affine_i, shape_i)  # [1, 4]
            logits.append(out_i)

        logits = torch.cat(logits, dim=0)  # [B, 4]
        loss = F.cross_entropy(logits, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (logits.argmax(dim=1) == label).sum().item()
        total += label.size(0)

    acc = correct / total
    print(f"[Fold {args.fold}] Epoch {epoch+1}: Train Loss={total_loss:.3f}, Train Acc={acc:.3f}")

    # ✅ 验证
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for batch in val_loader:
            img = batch['image'].to(device)
            label = batch['label'].to(device)
            affine = batch['affine']
            shape = batch['original_shape']

            logits = []
            for i in range(img.shape[0]):
                img_i = img[i].unsqueeze(0)
                affine_i = affine[i].to(device)
                shape_i = tuple(shape[i].tolist())  # ✅ 每个样本单独处理
                out_i = model(img_i, affine_i, shape_i)
                logits.append(out_i)
            logits = torch.cat(logits, dim=0)

            val_correct += (logits.argmax(dim=1) == label).sum().item()
            val_total += label.size(0)

        val_acc = val_correct / val_total
        print(f"[Fold {args.fold}] Epoch {epoch + 1}: Val Acc={val_acc:.3f}")

