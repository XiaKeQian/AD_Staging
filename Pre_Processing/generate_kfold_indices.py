import pickle
from sklearn.model_selection import StratifiedKFold
import os

# === 设置参数 ===
INDEX_PATH = "../data/labels/index.pkl"  # 原始索引文件
OUTPUT_DIR = "../data/labels/kfold"      # 输出目录
K = 5                                 # K折交叉验证

# === 创建目录 ===
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 加载索引 ===
with open(INDEX_PATH, 'rb') as f:
    index_dict = pickle.load(f)  # {ptid: (path, label)}

items = list(index_dict.items())          # [(ptid, (path, label))]
paths = [v[0] for k, v in items]          # path
labels = [v[1] for k, v in items]         # label

skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(paths, labels)):
    train_dict = {items[i][0]: items[i][1] for i in train_idx}
    val_dict   = {items[i][0]: items[i][1] for i in val_idx}

    with open(f"{OUTPUT_DIR}/train_fold{fold}.pkl", 'wb') as f:
        pickle.dump(train_dict, f)
    with open(f"{OUTPUT_DIR}/val_fold{fold}.pkl", 'wb') as f:
        pickle.dump(val_dict, f)

    print(f"[Fold {fold}] 训练集: {len(train_dict)} 条, 验证集: {len(val_dict)} 条 ✅")

print("✅ 所有 fold 生成完毕！")
