import os
import pandas as pd
import pickle

# === 路径设置 ===
CSV_PATH = "/home/xiakeqian/Desktop/ADNI1_Baseline_Final_Label_with_MRI_Check.csv"
ADNI_ROOT = "/home/xiakeqian/Desktop/HOPE-for-mild-cognitive-impairment/data/ADNI"  # 根目录，下面是每个 PTID 文件夹
OUTPUT_PKL = "index.pkl"

# === 标签映射 ===
label_map = {
    'CN': 0,
    'AD': 1,
    'sMCI': 2,
    'pMCI': 3
}

# === 读取 CSV 并筛选合法样本 ===
df = pd.read_csv(CSV_PATH)
df = df[(df['Final'].isin(label_map)) & (df['MRI_Available'] == True)]

# === 遍历每个 PTID，递归查找 aligned.npy ===
index_dict = {}

for _, row in df.iterrows():
    ptid = row['PTID']
    label = label_map[row['Final']]

    ptid_root = os.path.join(ADNI_ROOT, ptid)
    found = None

    for root, _, files in os.walk(ptid_root):
        if 'aligned.npy' in files:
            found = os.path.join(root, 'aligned.npy')
            break

    if found:
        index_dict[ptid] = (found, label)
    else:
        print(f"[⚠️ WARNING] 没找到 {ptid} 的 aligned.npy 文件。")

# === 保存为 .pkl ===
with open(OUTPUT_PKL, 'wb') as f:
    pickle.dump(index_dict, f)

print(f"✅ 索引文件生成成功，共 {len(index_dict)} 条记录。")
print(f"保存位置: {OUTPUT_PKL}")
