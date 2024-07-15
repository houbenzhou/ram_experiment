import json
import os
import time
from pathlib import Path
import cv2
import faiss
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np

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
def get_image_color_features(image_path, image_size=(16, 16)):
    """
    创建直方图
    :param template_feature:模板以及当前帧的多个bbox的roi
    :return:
    """
    # image = cv2.imread(image_path)
    # OpenCV读取中文路径
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    image = cv2.resize(image, image_size)

    '''先进行HSV做亮度V的均衡'''
    image = handle_img(image)
    """"创建 RGB 三通道直方图（直方图矩阵）"""
    h, w, c = image.shape
    # 创建一个（16*16*16,1）的初始矩阵，作为直方图矩阵
    # 16*16*16的意思为三通道每通道有16个bins
    # rgbhist = np.zeros([16 * 16 * 16, 1], np.float32)
    # bsize = 256 / 16
    b_array = []
    g_array = []
    r_array = []
    for row in range(h):
        for col in range(w):
            b = image[row, col, 0]
            g = image[row, col, 1]
            r = image[row, col, 2]
            b_array.append(b/255)
            g_array.append(g/255)
            r_array.append(r/255)
            # 人为构建直方图矩阵的索引，该索引是通过每一个像素点的三通道值进行构建
    image_features=[]
    image_features.append(np.concatenate((b_array, g_array, r_array)).astype('float32'))
    image_features=np.array(image_features)
    image_features=image_features-0.5
    cv2.destroyAllWindows()
    return image_features

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
    # 均值方差归一化
    mean = np.mean(image_features)
    std = np.std(image_features)
    image_features = (image_features - mean)/std
    # 关闭图像，释放资源
    image.close()
    return image_features
def image_search(img_path, k=1):
    # image = Image.open(img_path)

    image_features_color = get_image_color_features(img_path)
    image_features_clip = get_image_clip_features(img_path)
    image_features = np.concatenate((image_features_clip, image_features_color), axis=1)

    D, I = index.search(image_features, k)  # 实际的查询

    filenames = [[id2filename[str(j)] for j in i] for i in I]

    return img_path, D, filenames


if __name__ == "__main__":
    base_path = os.getcwd()
    clip_model_path = os.path.join(base_path, "model","clip_model")
    faiss_path = os.path.join(base_path,"output","faiss_model","color_clip_feature","clean_data_5037")
    img_path = os.path.join(base_path, "data", "clean_data_5037")

    # 加载clip模型
    clip_model = CLIPModel.from_pretrained(clip_model_path)
    clip_processor = CLIPProcessor.from_pretrained(clip_model_path)
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
            img_path, D, filenames = image_search(img_path, k=4)
            true_name = extract_directory_name(filenames[0][0],-2)
            pre_name = extract_directory_name(filenames[0][1],-2)
            total_count=total_count+1
            for i in range(len(filenames)):
                if true_name == pre_name:
                    correct_count = correct_count + 1
                    correct_file.write(f"img_path: {img_path} pred_name: {filenames[0][1]}" + "\n")
                else:
                    error_file.write( f"img_path: {img_path} pred_name: {filenames[0][1]}"+"\n")
    with open(out_correct_rate_logs, 'w', encoding='utf-8') as correct_rate:
        correct_rate.write(f"correct_count:{correct_count}, total_count: {total_count} ,accuracy: {correct_count/total_count}."+"\n")



