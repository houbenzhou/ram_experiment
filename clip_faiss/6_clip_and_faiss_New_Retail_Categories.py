import os

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, CLIPFeatureExtractor
import json
import faiss
from tqdm  import tqdm
proxy_url = "http://192.168.0.30:10809"
os.environ['HTTP_PROXY'] = proxy_url
os.environ['HTTPS_PROXY'] = proxy_url
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
from pathlib import Path


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
    processed = processor(images=image, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        image_features = model.get_image_features(pixel_values=processed["pixel_values"])
    return image_features


def get_text_feature(text: str):
    processed = processor(text=text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        text_features = model.get_text_features(processed['input_ids'])
    return text_features


d = 512
index = faiss.IndexFlatL2(d)  # 使用 L2 距离

# folder_path = r"./data/Trax_bbox出来的小图含label_20230207"
# folder_path = os.path.join(r"./data/新零售图片数据_Trax_部分")
base_path=os.getcwd()
folder_path = os.path.join(base_path,'data','Trax_bbox出来的小图含label_20230207')
# 遍历文件夹
file_paths = []
for root, dirs, files in os.walk(folder_path):
    for file in files:
        # 检查文件是否为图片文件(这里简单地检查文件扩展名)
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            file_path = os.path.join(root, file)
            file_paths.append(file_path)

id2filename = {idx: extract_directory_name(x,-2) for idx, x in enumerate(file_paths)}

# 保存为 JSON 文件
with open('./output_all/id2filename.json', 'w') as json_file:
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

faiss.write_index(index, "./output_all/image.faiss")



