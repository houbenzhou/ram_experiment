import os

import cv2
import faiss
import numpy as np
from PIL import Image
import os

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, CLIPFeatureExtractor, AutoImageProcessor, AutoModel
import json
import faiss
from tqdm import tqdm
from pathlib import Path

# 将图像缩放至同一尺寸
# def resize(image_path, image_size=(256, 256)):
#     img = Image.open(image_path)
#     pass
def resize(image_path, image_size=(256, 256)):
    img = cv2.imread(image_path)
    img = cv2.resize(img, image_size)
    # img = Image.fromarray(img).convert("RGB")
    return img
#创建颜色直方图
def get_dinov2_features(image_path):
    """

    :return:
    """
    image = Image.open(image_path)
    inputs = dinov2_processor(images=image, return_tensors="pt")
    outputs = dinov2_model(**inputs)
    image_features = outputs.last_hidden_state
    image_features = image_features.mean(dim=1)
    image_features = image_features.detach().numpy()
    # image_features = outputs.last_hidden_state.detach().numpy()
    return image_features

def handle_img(img):
    """
    对HSV颜色空间中的V（亮度做一下均衡)
    :param img:均衡后的图像，再转回BGR
    :return:
    """
    # img = cv2.resize(img, (100, 100))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 对HSV颜色空间中的V（亮度做一下均衡），再转会BGR
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    return img


def create_index(vector_dim, vectors):
    # 创建一个索引
    index = faiss.IndexFlatL2(vector_dim)  # 使用L2距离

    # 添加向量到索引
    if isinstance(vectors, np.ndarray):
        index.add(vectors)
    else:
        index.add(np.array(vectors, dtype='float32'))

    return index

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

# 示例：使用颜色直方图创建FAISS索引
if __name__ == "__main__":
    base_path = os.getcwd()
    # 图片路径
    image_path = os.path.join(base_path, 'data', 'clean_data_5037')
    # 向量长度
    d = 768
    # 输出文件名
    out_name = 'clean_data_5037'

    # dinov2_model的模型路径
    dinov2_model_path = os.path.join(base_path, "model", "dinov2_model")
    id_type = 'image_path'

    '''
    生成图片faiss索引文件的路径
        category_name:类别名称从文件夹路径中获取
        image_path:类别名称是图片路径
    '''

    if id_type == 'category_name':
        out_path = os.path.join(base_path, 'output', 'faiss_model', 'dinov2_feature',out_name)
    elif id_type == 'image_path':
        out_path = os.path.join(base_path, 'output', 'faiss_model', 'dinov2_feature',out_name)

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    out_category_name = os.path.join(out_path, 'category_name.json')
    out_image_faiss=os.path.join(out_path, 'image_faiss.index')
    # 加载dinov2模型
    dinov2_processor= AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    dinov2_model = AutoModel.from_pretrained("facebook/dinov2-base")

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
        image_features = get_dinov2_features(file_path)

        index.add(image_features)


    faiss.write_index(index, out_image_faiss)


