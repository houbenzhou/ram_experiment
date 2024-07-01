
'''
    CLIP模型能把文本和图片都变成同一个空间里面的向量。而且，文本和图片之间还有关联.
 我们是不是可以利用这个向量来进行语义检索，实现搜索图片的功能？答案是可以的。

'''

###加载数据集
from datasets import load_dataset
import os
proxy_url="http://192.168.0.30:10809"
os.environ['HTTP_PROXY']=proxy_url
os.environ['HTTPS_PROXY']=proxy_url
dataset = load_dataset("rajuptvs/ecommerce_products_clip")
dataset

import matplotlib.pyplot as plt

training_split = dataset["train"]

## 显示图片
def display_images(images):
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()

    for idx, img in enumerate(images):
        axes[idx].imshow(img)
        axes[idx].axis('off')

    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    plt.show()

images = [example["image"] for example in training_split.select(range(10))]
# display_images(images)

## 利用CLIP模型把所有的图片都转换成向量并记录下来。获取向量的方法和我们做零样本分类类似，
# 我们加载了CLIPModel和CLIPProcessor，通过get_image_feature()函数把图片转换成向量,
# 再通过add_image_features()函数把向量添加到features中。
# 我们一条记录一条记录地来处理训练集里面的图片特征，并且把处理完成的特征也加入到数据集的features属性里面去。

import torch
import torchvision.transforms as transforms
from PIL import Image
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_image_features(image):
    with torch.no_grad():
        inputs = processor(images=[image], return_tensors="pt", padding=True)
        inputs.to(device)
        features = model.get_image_features(**inputs)
    return features.cpu().numpy()

def add_image_features(example):
    example["features"] = get_image_features(example["image"])
    return example

# Apply the function to the training_split
training_split = training_split.map(add_image_features)

## 有了处理好的向量，把这些向量都放到Faiss的索引里面去。
import numpy as np
import faiss

features = [example["features"] for example in training_split]
features_matrix = np.vstack(features)

dimension = features_matrix.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(features_matrix.astype('float32'))

## 如果搜索图片呢？
'''
    首先，get_text_features 这个函数会通过 CLIPModel 和 CLIPProcessor 拿到一段文本输入的向量。
    其次是 search 函数。它接收一段搜索文本，
    然后将文本通过 get_text_features 转换成向量，去 Faiss 里面搜索对应的向量索引。
    然后通过这个索引重新从 training_split 里面找到对应的图片，加入到返回结果里面去。
    然后我们就以 A red dress 作为搜索词，调用 search 函数拿到搜索结果。
    最后，我们通过 display_search_results 这个函数，将搜索到的图片以及在 Faiss 索引中的距离展示出来。
'''


def get_image_features(image_path):
    # Load the image from the file
    image = Image.open(image_path).convert("RGB")

    with torch.no_grad():
        inputs = processor(images=[image], return_tensors="pt", padding=True)
        inputs.to(device)
        features = model.get_image_features(**inputs)
    return features.cpu().numpy()


def search(image_path, top_k=5):
    # Get the image feature vector for the input image
    image_features = get_image_features(image_path)

    # Perform a search using the FAISS index
    distances, indices = index.search(image_features.astype("float32"), top_k)

    # Get the corresponding images and distances
    results = [
        {"image": training_split[i.item()]["image"], "distance": distances[0][j]}
        for j, i in enumerate(indices[0])
    ]

    return results


image_path = "./data/cat1.png"
results = search(image_path)
print(results)



