import random
import time

import torch
from transformers import CLIPProcessor, CLIPModel, CLIPFeatureExtractor

import faiss
from PIL import Image
import os
import json
from collections import Counter
proxy_url = "http://192.168.0.30:10809"
os.environ['HTTP_PROXY'] = proxy_url
os.environ['HTTPS_PROXY'] = proxy_url
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
# 保存为 JSON 文件
with open('./output/id2filename.json', 'r') as json_file:
    id2filename = json.load(json_file)
index = faiss.read_index("./output/image.faiss")

def most_frequent_element(arr):
    if not arr:
        return None  # 如果数组为空，则返回None
    element_count = Counter(arr)
    # 获取出现次数最多的元素和它的次数
    most_common_element, frequency = element_count.most_common(1)[0]
    return most_common_element, frequency

def get_random_image_path(directory):
    # 存储找到的图片路径
    image_paths = []

    # os.walk遍历目录和子目录
    for root, dirs, files in os.walk(directory):
        for file in files:
            # 检查文件是否是图片，这里以几种常见图片格式为例
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_paths.append(os.path.join(root, file))

    # 从列表中随机选择一个图片路径
    if image_paths:
        return random.choice(image_paths)
    else:
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
    processed = processor(images=image, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        image_features = model.get_image_features(pixel_values=processed["pixel_values"])
    return image_features
def get_text_feature(text: str):
    processed = processor(text=text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        text_features = model.get_text_features(processed['input_ids'])
    return text_features





def text_search(text, k=1):
    inputs = processor(text=text, images=None, return_tensors="pt", padding=True)
    text_features = model.get_text_features(inputs["input_ids"], inputs["attention_mask"])
    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)  # normalize
    text_features = text_features.detach().numpy()
    D, I = index.search(text_features, k)  # 实际的查询

    filenames = [[id2filename[str(j)] for j in i] for i in I]

    return text, D, filenames


def image_search(image, k=1):

    inputs = processor(images=image, return_tensors="pt")
    image_features = model.get_image_features(**inputs)
    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)  # normalize

    image_features = image_features.detach().numpy()
    D, I = index.search(image_features, k)  # 实际的查询

    filenames = [[id2filename[str(j)] for j in i] for i in I]

    return img_path, D, filenames


if __name__ == "__main__":
    # text = ["drinks"]
    # text, D, filenames = text_search(text)
    # print(text, D, filenames)
    # img_path = "./data/新零售图片数据_Trax_部分/嗡嗡乐枇杷蜂蜜500G/1674952677728_2_2231.0_2097.5_434.0_591.0_0.0.jpg"
    # # img_path = "./data/新零售图片数据_Trax_部分/(复)特慧优苏打饼干(奶盐味)388G/1674955196040_0_2918.0_3612.0_1108.0_494.0_0.0.jpg"
    # # img_path = "./data/新零售图片数据_Trax_部分/(复)特慧优牛轧糖(蔓越莓味)300G/1674957627488_2_1853.5_3519.5_325.0_951.0_0.0.jpg"
    # start_time=time.time()
    # img_path,D,filenames = image_search(img_path,k=4)
    # end_time=time.time()
    # print("cost_time:",end_time-start_time)
    # # print(img_path,D,filenames)
    # print(filenames)
    base_images=os.getcwd()
    img_path = os.path.join(base_images,"data","新零售图片数据_Trax_部分")
    img_paths=get_image_path(img_path)
    # 打印查询结果
    # for img_path in img_paths:
    #     image = Image.open(img_path)
    #     start_time=time.time()
    #     img_path,D,filenames = image_search(image,k=4)
    #     end_time=time.time()
    #     print("cost_time:",end_time-start_time)
    #     # print(img_path,D,filenames)
    #     print(filenames)
    # 统计正确分类总个数，图片总个数，正确率，按照查询类别最优指标来计算

    # correct_count = 0
    # total_count = 0
    # for img_path in img_paths:
    #     image = Image.open(img_path)
    #     start_time = time.time()
    #     img_path, D, filenames = image_search(image, k=4)
    #     total_count=total_count+1
    #     for i in range(len(filenames)):
    #         if filenames[0][0] == filenames[0][1]:
    #             correct_count = correct_count + 1
    #         else:
    #             print("img_path:", img_path, "true_name:", filenames[0][0], "pred_name:", filenames[0][1])
    #
    # print("correct_count:",correct_count,"total_count:",total_count,"accuracy:",correct_count/total_count)


    correct_count = 0
    total_count = 0
    for img_path in img_paths:
        image = Image.open(img_path)
        start_time = time.time()
        img_path, D, filenames = image_search(image, k=4)
        temp_filenames=filenames[0]
        true_name=filenames[0][0]
        temp_filenames.pop(0)
        total_count=total_count+1
        pred_name, freq = most_frequent_element(temp_filenames)
        # print("true_name:",true_name,"pred_name:",pred_name,"freq:",freq)
        print(D)
        if true_name == pred_name:
            correct_count = correct_count + 1
        else:
            print("img_path:", img_path, "true_name:", true_name, "pred_name:", pred_name, "freq:", freq)

    print("correct_count:",correct_count,"total_count:",total_count,"accuracy:",correct_count/total_count)

