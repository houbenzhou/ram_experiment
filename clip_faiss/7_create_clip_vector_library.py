import os

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, CLIPFeatureExtractor
import json
import faiss
from tqdm import tqdm
from pathlib import Path
import numpy as np

"""
    创建faiss数据库，文件夹下包含两个文件：
    category_name.json：从文件夹目录名称获取类别名称
    image.faiss：从文件夹中获取图片特征向量

"""
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

def get_image_feature(filename: str):
    image = Image.open(filename).convert("RGB")
    processed = clip_processor(images=image, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        image_features = clip_model.get_image_features(pixel_values=processed["pixel_values"])
    return image_features

def get_image_clip_features(file_path):
    '''
        获取图像的clip特征
    '''
    image = Image.open(file_path)
    inputs = clip_processor(images=image, return_tensors="pt", padding=True)
    image_features = clip_model.get_image_features(inputs["pixel_values"])
    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)  # normalize
    image_features = image_features.detach().numpy()
    # 最大最小值归一化
    # min_image_features = np.min(image_features)
    # max_image_features = np.max(image_features)
    # image_features = (image_features - min_image_features)

    # 均值方差归一化
    mean = np.mean(image_features)
    std = np.std(image_features)
    image_features = (image_features - mean)/std
    # 关闭图像，释放资源
    image.close()
    return image_features


if __name__ == '__main__':
    base_path = os.getcwd()
    # 图片路径
    image_path = os.path.join(base_path, 'data', 'clean_data_5037_correct_2')
    # 向量长度
    d = 512
    # 输出文件名
    out_name = 'clean_data_5037_correct_2'

    # clip_model的模型路径
    clip_model_path = os.path.join(base_path, "model", "clip_model")
    id_type = 'image_path'
    '''
    生成图片faiss索引文件的路径
        category_name:类别名称从文件夹路径中获取
        image_path:类别名称是图片路径
    '''

    if id_type == 'category_name':
        out_path = os.path.join(base_path, 'output', 'faiss_model', 'clip',out_name)
    elif id_type == 'image_path':
        out_path = os.path.join(base_path, 'output', 'faiss_model','clip', out_name)


    if not os.path.exists(out_path):
        os.makedirs(out_path)
    out_category_name = os.path.join(out_path, 'category_name.json')
    out_image_faiss=os.path.join(out_path, 'image_faiss.index')
    # 加载clip模型文件
    clip_model = CLIPModel.from_pretrained(clip_model_path)
    clip_processor = CLIPProcessor.from_pretrained(clip_model_path)

    index = faiss.IndexFlatL2(d)  # 使用 L2 距离
    # 遍历文件夹
    file_paths = []
    for root, dirs, files in os.walk(image_path):
        for file in files:
            # 检查文件是否为图片文件(这里简单地检查文件扩展名)
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                file_path = os.path.join(root, file)
                file_paths.append(file_path)

    if id_type == 'category_name':
        id2filename = {idx: extract_directory_name(x,-2) for idx, x in enumerate(file_paths)}
    elif id_type == 'image_path':
        id2filename = {idx: x for idx, x in enumerate(file_paths)}

    # 保存为 JSON 文件
    with open(out_category_name, 'w') as json_file:
        json.dump(id2filename, json_file)

    for file_path in tqdm(file_paths, total=len(file_paths)):
        # 使用PIL打开图片
        image_features=get_image_clip_features(file_path)
        index.add(image_features)

    faiss.write_index(index, out_image_faiss)



