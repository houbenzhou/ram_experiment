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
def get_image_path(directory):
    # 存储找到的图片路径
    image_paths = []

    # os.walk遍历目录和子目录
    for root, dirs, files in os.walk(directory):
        for file in files:
            # 检查文件是否是图片，这里以几种常见图片格式为例
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_paths.append(os.path.join(root, file))

    return image_paths
def extract_directory_name(path, level):
    """
    Extracts a directory name from a given path at the specified level.

    :param path: The file path from which to extract the directory.
    :param level: The level of the directory to extract, where 0 is the top-level.
    :return: The name of the directory at the specified level or None if level is out of range.
    """
    try:
        # 将字符串路径转换为Path对象
        p = Path(path)
        # 通过parts属性获取所有部分的元组，筛选出目录部分
        # 忽略最后一部分如果它是文件名
        directories=p.parts
        # directories = [part for part in p.parts if Path(part).is_dir()]
        # 返回指定层级的目录名
        return directories[level]
    except IndexError:
        # 如果指定的层级不存在，返回None
        print(path)
        return None


import os


def check_files_in_directory(directory_path):
    # 检查目录路径是否存在
    if not os.path.exists(directory_path):
        return "指定的目录不存在。"

    # os.listdir列出目录中的所有文件和文件夹
    files = os.listdir(directory_path)

    # # 筛选出文件（排除文件夹）
    # files = [file for file in files if os.path.isfile(os.path.join(directory_path, file))]
    #
    # if files:
    #     return f"目录中有文件。具体文件数: {len(files)}"
    # else:
    #     return "目录中没有文件。"
    return files


import os
import shutil


def copy_files(src, dest):
    shutil.copy(src, dest)





def extract_directory_name(path, level):
    """
    Extracts a directory name from a given path at the specified level.

    :param path: The file path from which to extract the directory.
    :param level: The level of the directory to extract, where 0 is the top-level.
    :return: The name of the directory at the specified level or None if level is out of range.
    """
    try:
        # 将字符串路径转换为Path对象
        p = Path(path)
        # 通过parts属性获取所有部分的元组，筛选出目录部分
        # 忽略最后一部分如果它是文件名
        directories=p.parts
        # directories = [part for part in p.parts if Path(part).is_dir()]
        # 返回指定层级的目录名
        return directories[level]
    except IndexError:
        # 如果指定的层级不存在，返回None
        return None


def get_image_path(directory):
    # 存储找到的图片路径
    image_paths = []

    # os.walk遍历目录和子目录
    for root, dirs, files in os.walk(directory):
        for file in files:
            # 检查文件是否是图片，这里以几种常见图片格式为例
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_paths.append(os.path.join(root, file))

    # 从列表中随机选择一个图片路径
    return image_paths

def get_image_feature(filename: str):
    image = Image.open(filename).convert("RGB")
    processed = clip_processor(images=image, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        image_features = clip_model.get_image_features(pixel_values=processed["pixel_values"])
    return image_features





def image_search(image, k=1):

    inputs = clip_processor(images=image, return_tensors="pt")
    image_features = clip_model.get_image_features(**inputs)
    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)  # normalize

    image_features = image_features.detach().numpy()
    D, I = index.search(image_features, k)  # 实际的查询

    filenames = [[id2filename[str(j)] for j in i] for i in I]

    return img_path, D, filenames



if __name__ == "__main__":
    base_path = os.getcwd()
    clip_model_path = os.path.join(base_path,"..", "model","clip_model")
    faiss_path = os.path.join(base_path,"..","output","faiss_model","35_image_path")
    img_path = os.path.join(base_path,"..", "data", "新零售图片数据_Trax_部分")
    out_img_path = os.path.join(base_path,"..", "data", "35_image_path_corrcet")


    # 加载clip模型
    clip_model = CLIPModel.from_pretrained(clip_model_path)
    clip_processor = CLIPProcessor.from_pretrained(clip_model_path)
    # 加载faiss索引
    with open(os.path.join(faiss_path,"category_name.json"), 'r') as json_file:
        id2filename = json.load(json_file)
    index = faiss.read_index(os.path.join(faiss_path,"image_faiss.index"))

    img_paths=get_image_path(img_path)
    out_error_picture_name_logs = os.path.join(faiss_path,"correct_picture_name.txt")
    # out_correct_rate_logs = os.path.join(faiss_path,"correct_rate.txt")


    for img_path in img_paths:
        image = Image.open(img_path)
        start_time = time.time()
        img_path, D, filenames = image_search(image, k=4)
        true_name = extract_directory_name(filenames[0][0],-2)
        pre_name = extract_directory_name(filenames[0][1],-2)

        if true_name == pre_name:
            out_sub_img_path=os.path.join(out_img_path,true_name)
            if not os.path.exists(out_sub_img_path):
                os.makedirs(out_sub_img_path)

            img_filename=os.path.basename(filenames[0][0])
            copy_files(filenames[0][0],os.path.join(out_sub_img_path,img_filename))





