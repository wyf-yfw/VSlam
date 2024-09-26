# # 创建并打开一个新的 txt 文件
# for i in range(1500, 2000):
#     txt_name = "/Users/wanggang/PycharmProjects/slam/dataset/label/val/triangle/triangle-{}.txt".format(i)
#     print(txt_name)
#     with open(txt_name, 'w') as file:
#         # 写入内容
#         file.write('7')
import os
import shutil

# 源文件夹和目标文件夹路径
source_folder = '/Users/wanggang/PycharmProjects/slam/dataset/val/labels/circle'  # 替换为你的源文件夹路径
destination_folder = '/Users/wanggang/PycharmProjects/slam/dataset/val/labels'  # 替换为你的目标文件夹路径

# 确保目标文件夹存在
os.makedirs(destination_folder, exist_ok=True)

# 遍历源文件夹中的所有文件
for filename in os.listdir(source_folder):
    if filename.endswith('.txt'):
        # 构建完整的源文件路径和目标文件路径
        src_path = os.path.join(source_folder, filename)
        dst_path = os.path.join(destination_folder, filename)

        # 移动文件
        shutil.move(src_path, dst_path)
        print(f'Moved: {src_path} to {dst_path}')

print('所有 JPG 文件已成功移动。')
