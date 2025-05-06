# from nilearn import datasets
#
# atlas = datasets.fetch_atlas_aal()
# print(atlas.keys())  # 会看到 'maps', 'labels'
#
# # 保存为文件
# from nibabel import load, save
# save(load(atlas['maps']), 'AAL2.nii.gz')

from nilearn import plotting
import nibabel as nib

# 加载 AAL 图谱
img = nib.load("AAL2.nii.gz")

# 显示一个切片图（中轴面 sagittal）
plotting.plot_roi(img, display_mode='ortho', title='AAL atlas')

plotting.show()