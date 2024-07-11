import os

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, CLIPFeatureExtractor
import json
import faiss
from tqdm import tqdm
from pathlib import Path

"""
    创建faiss数据库，文件夹下包含两个文件：
    category_name.json：从文件夹目录名称获取类别名称
    image.faiss：从文件夹中获取图片特征向量

"""

def get_image_feature(filename: str):
    image = Image.open(filename).convert("RGB")
    processed = processor(images=image, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        image_features = model.get_image_features(pixel_values=processed["pixel_values"])
    return image_features


if __name__ == '__main__':
    base_path = os.getcwd()
    # clip_model的模型路径
    clip_model_path = os.path.join(base_path, "model", "clip_model")

    # # 35类用于生成图片特征的原始图像数据库，数据库文件组织路径为文件夹是类别名称，文件是图片
    # image_path = os.path.join(base_path, 'data', '新零售图片数据_Trax_部分')
    # id_type = 'image_path'
    # '''
    # 生成图片faiss索引文件的路径
    #     category_name:类别名称从文件夹路径中获取
    #     image_path:类别名称是图片路径
    # '''
    #
    # if id_type == 'category_name':
    #     out_path = os.path.join(base_path, 'output', 'faiss_model', '35_category_name')
    # elif id_type == 'image_path':
    #     out_path=os.path.join(base_path, 'output','faiss_model','35_image_path')

    # 5037类用于生成图片特征的原始图像数据库，数据库文件组织路径为文件夹是类别名称，文件是图片
    # image_path = os.path.join(base_path, 'data', 'Trax_bbox出来的小图含label_20230207')
    # id_type = 'category_name'
    # # 生成图片faiss索引文件的路径
    # # category_name:类别名称从文件夹路径中获取
    # # image_path:类别名称是图片路径
    # '''
    #  生成图片faiss索引文件的路径
    #      category_name:类别名称从文件夹路径中获取
    #      image_path:类别名称是图片路径
    # '''
    # if id_type == 'category_name':
    #     out_path = os.path.join(base_path, 'output', 'faiss_model', '5037_category_name')
    # elif id_type == 'image_path':
    #     out_path=os.path.join(base_path, 'output','faiss_model','5037_image_path')

    # # 5000数据清洗过的类用于生成图片特征的原始图像数据库，数据库文件组织路径为文件夹是类别名称，文件是图片
    # image_path = os.path.join(base_path, 'data', 'out_clean_data')
    # id_type = 'image_path'
    # '''
    # 生成图片faiss索引文件的路径
    #     category_name:类别名称从文件夹路径中获取
    #     image_path:类别名称是图片路径
    # '''
    #
    # if id_type == 'category_name':
    #     out_path = os.path.join(base_path, 'output', 'faiss_model', 'out_clean_data_image_path')
    # elif id_type == 'image_path':
    #     out_path=os.path.join(base_path, 'output','faiss_model','out_clean_data_image_path')
    # # 5000数据清洗过且裁切10%的类用于生成图片特征的原始图像数据库，数据库文件组织路径为文件夹是类别名称，文件是图片
    # image_path = os.path.join(base_path, 'data', 'out_clean_data_cropped_images_10')
    # id_type = 'image_path'
    # '''
    # 生成图片faiss索引文件的路径
    #     category_name:类别名称从文件夹路径中获取
    #     image_path:类别名称是图片路径
    # '''
    #
    # if id_type == 'category_name':
    #     out_path = os.path.join(base_path, 'output', 'faiss_model', 'out_clean_10_data_image_path')
    # elif id_type == 'image_path':
    #     out_path=os.path.join(base_path, 'output','faiss_model','out_clean_10_data_image_path')

    # 5000数据清洗过且裁切10%的类用于生成图片特征的原始图像数据库，数据库文件组织路径为文件夹是类别名称，文件是图片
    image_path = os.path.join(base_path, 'data', 'clean_data_5037_correct_2')
    id_type = 'image_path'
    '''
    生成图片faiss索引文件的路径
        category_name:类别名称从文件夹路径中获取
        image_path:类别名称是图片路径
    '''

    if id_type == 'category_name':
        out_path = os.path.join(base_path, 'output', 'faiss_model', 'clean_data_5037_correct_2')
    elif id_type == 'image_path':
        out_path = os.path.join(base_path, 'output', 'faiss_model', 'clean_data_5037_correct_2')


    if not os.path.exists(out_path):
        os.makedirs(out_path)
    out_category_name = os.path.join(out_path, 'category_name.json')
    out_image_faiss=os.path.join(out_path, 'image_faiss.index')
    # 加载clip模型文件
    model = CLIPModel.from_pretrained(clip_model_path)
    processor = CLIPProcessor.from_pretrained(clip_model_path)

    d = 512
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
        image = Image.open(file_path)
        inputs = processor(images=image, return_tensors="pt", padding=True)
        image_features = model.get_image_features(inputs["pixel_values"])
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)  # normalize
        image_features = image_features.detach().numpy()

        index.add(image_features)
        # 关闭图像，释放资源
        image.close()

    faiss.write_index(index, out_image_faiss)



