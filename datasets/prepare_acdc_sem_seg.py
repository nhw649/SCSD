import os
import shutil

# 父目录路径
parent_dir = '/data/nhw/datasets/DG/acdc/gt'

# 获取父目录中的所有子目录
sub_dirs = [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]
print(sub_dirs)
# # 遍历每个子目录
for sub_dir in sub_dirs:
    val_dir_path = os.path.join(parent_dir, sub_dir, 'val')
    inner_sub_dirs = [d for d in os.listdir(val_dir_path) if os.path.isdir(os.path.join(val_dir_path, d))]
    print(inner_sub_dirs)
    for inner_sub_dir in inner_sub_dirs:
        inner_dir_path = os.path.join(val_dir_path, inner_sub_dir)
        print(inner_dir_path)
        # 获取子目录中的所有文件
        files = os.listdir(inner_dir_path)

        for file_name in files:
            file_path = os.path.join(inner_dir_path, file_name)

            # 检查是否是文件
            if os.path.isfile(file_path):
                # 移动文件到val目录
                shutil.move(file_path, val_dir_path)

# print("文件提取完成。")