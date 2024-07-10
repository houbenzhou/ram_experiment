import os
from pathlib import Path
import json
import os
import time
from pathlib import Path

import faiss
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import shutil
import os
from PIL import Image
import os
import shutil

def copy_folders_with_two_or_more_images(source_directory, target_directory,file_num_threshold):
    # 检查目标文件夹是否存在，如果不存在则创建
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    # 遍历源目录中的所有子目录
    for folder in os.listdir(source_directory):
        folder_path = os.path.join(source_directory, folder)

        # 确保当前路径是一个目录
        if os.path.isdir(folder_path):
            image_files = [file for file in os.listdir(folder_path) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

            # 检查图像文件数量是否大于等于file_num_threshold
            if len(image_files) >= file_num_threshold:
                # 创建目标文件夹
                target_folder_path = os.path.join(target_directory, folder)
                if not os.path.exists(target_folder_path):
                    os.makedirs(target_folder_path)

                # 拷贝所有图像文件到目标文件夹
                for image_file in image_files:
                    src_file_path = os.path.join(folder_path, image_file)
                    dst_file_path = os.path.join(target_folder_path, image_file)
                    shutil.copy2(src_file_path, dst_file_path)
                print(f"Copied {len(image_files)} images to {target_folder_path}")

## 使用示例
if __name__ == '__main__':
    base_path = os.getcwd()
    source_directory = os.path.join(base_path, "data", "clean_data_5037_correct_2")
    target_directory = os.path.join(base_path, "data", "clean_data_5037_correct_3")
    copy_folders_with_two_or_more_images(source_directory, target_directory,file_num_threshold=3)

