import json
import os
import time
from pathlib import Path
import cv2
import faiss
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, AutoImageProcessor, AutoModel

from clip_faiss.toolkit import view_bar
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

def get_image_clip_features(file_path):
    '''
        获取图像的clip特征
    '''
    image = Image.open(file_path)
    inputs = clip_processor(images=image, return_tensors="pt", padding=True).to(device)
    image_features = clip_model.get_image_features(inputs["pixel_values"])
    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)  # normalize
    image_features = image_features.to("cpu")
    image_features = image_features.detach().numpy()
    # 均值方差归一化
    # mean = np.mean(image_features)
    # std = np.std(image_features)
    # image_features = (image_features - mean)/std
    # 关闭图像，释放资源
    image.close()
    return image_features

def image_search(image, k=1):


    image_features = get_image_clip_features(image)

    D, I = index.search(image_features, k)  # 实际的查询

    filenames = [[id2filename[str(j)] for j in i] for i in I]

    return img_path, D, filenames


if __name__ == "__main__":
    base_path = os.getcwd()
    # clip_model的模型路径
    clip_model_path = os.path.join(base_path, "..","model", "clip_model")
    faiss_path = os.path.join(base_path,"output","faiss_model","clip","35_image_path")
    img_path = os.path.join(base_path, "..","data", "新零售图片数据_Trax_部分")

    # 加载clip模型文件
    clip_model = CLIPModel.from_pretrained(clip_model_path,local_files_only=True).to(device)
    clip_processor = CLIPProcessor.from_pretrained(clip_model_path,local_files_only=True)

    # 加载faiss索引
    with open(os.path.join(faiss_path,"category_name.json"), 'r') as json_file:
        id2filename = json.load(json_file)
    index = faiss.read_index(os.path.join(faiss_path,"image_faiss.index"))

    img_paths=get_image_path(img_path)
    out_error_picture_name_logs = os.path.join(faiss_path,"error_picture_name.txt")
    out_correct_picture_name_logs = os.path.join(faiss_path,"correct_picture_name.txt")
    out_correct_rate_logs = os.path.join(faiss_path,"correct_rate.txt")

    # 打印查询结果
    correct_count = 0
    total_count = 0
    with open(out_error_picture_name_logs, 'w',encoding='utf-8') as error_file,open(out_correct_picture_name_logs, 'w',encoding='utf-8') as correct_file:
        for img_path in img_paths:
            start_time = time.time()
            img_path, D, filenames = image_search(img_path, k=10)
            end_time = time.time()
            true_name = extract_directory_name(filenames[0][0],-2)
            pre_name = extract_directory_name(filenames[0][1],-2)
            total_count=total_count+1
            view_bar(total_count,len(img_paths))
            print("cost_time:", end_time - start_time)
            for i in range(len(filenames)):
                if true_name == pre_name:
                    correct_count = correct_count + 1
                    correct_file.write(
                        f"img_path: {img_path} pred_name1: {filenames[0][1]} pred_name2: {filenames[0][2]} pred_name3: {filenames[0][3]} pred_1_vector: {D[0][1]} pred_2_vector: {D[0][2]} pred_2_vector: {D[0][3]} " + "\n")
                else:
                    error_file.write(
                        f"img_path: {img_path} pred_name1: {filenames[0][1]} pred_name2: {filenames[0][3]} pred_name3: {filenames[0][3]} pred_1_vector: {D[0][1]} pred_2_vector: {D[0][2]} pred_2_vector: {D[0][3]} " + "\n")
    with open(out_correct_rate_logs, 'w', encoding='utf-8') as correct_rate:
         correct_rate.write(f"correct_count:{correct_count}, total_count: {total_count} ,accuracy: {correct_count / total_count}." + "\n")
