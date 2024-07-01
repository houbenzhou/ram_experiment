import os

import torch
from PIL import Image
from IPython.display import display
from IPython.display import Image as IPyImage
from transformers import CLIPProcessor, CLIPModel,CLIPFeatureExtractor

proxy_url="http://192.168.0.30:10809"
os.environ['HTTP_PROXY']=proxy_url
os.environ['HTTPS_PROXY']=proxy_url
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

'''
计算文字与图片的相似度
    步骤一：通过Transformers库的CLIPModel和CLIPProcessor，加载了 clip-vit-base-patch32 这个模型，用来处理我们的图片和文本信息。
    步骤二：在get_image_features方法里，我们做了两件事情。
        首先，我们通过刚才拿到的 CLIPProcessor 对图片做预处理，变成一系列的数值特征表示的向量。这个预处理的过程，其实就是把原始的图片，变成一个个像素的 RGB 值；然后统一图片的尺寸，以及对于不规则的图片截取中间正方形的部分，最后做一下数值的归一化。具体的操作步骤，已经封装在 CLIPProcessor 里了，你可以不用关心。
        然后，我们再通过CLIPModel，把上面的数值向量，推断成一个表达了图片含义的张量（Tensor）。这里，你就把它当成是一个向量就好了。
    步骤三：同样的，get_text_features也是类似的，先把对应的文本通过CLIPProcessor转换成Token，然后再通过模型推断出表示文本的张量。
    步骤四：我们定义了一个cosine_similarity函数，用来计算张量之间的余弦相似度。
    步骤五：我们利用上面这些函数计算图片和文本之间的相似度。
    
'''

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

def cosine_similarity(tensor1, tensor2):
    tensor1_normalized = tensor1 / tensor1.norm(dim=-1, keepdim=True)
    tensor2_normalized = tensor2 / tensor2.norm(dim=-1, keepdim=True)
    return (tensor1_normalized * tensor2_normalized).sum(dim=-1)

image_tensor = get_image_feature("./data/cat.jpg")

cat_text = "This is a cat."
cat_text_tensor = get_text_feature(cat_text)

dog_text = "This is a dog."
dog_text_tensor = get_text_feature(dog_text)

display(IPyImage(filename='./data/cat.jpg'))

print("Similarity with cat : ", cosine_similarity(image_tensor, cat_text_tensor))
print("Similarity with dog : ", cosine_similarity(image_tensor, dog_text_tensor))



'''
计算图片与图片的相似度
    步骤一：通过Transformers库的CLIPModel和CLIPProcessor，加载了 clip-vit-base-patch32 这个模型，用来处理我们的图片。
    步骤二：在get_image_features方法里，我们做了两件事情。
        首先，我们通过刚才拿到的 CLIPProcessor 对图片做预处理，变成一系列的数值特征表示的向量。这个预处理的过程，其实就是把原始的图片，变成一个个像素的 RGB 值；然后统一图片的尺寸，以及对于不规则的图片截取中间正方形的部分，最后做一下数值的归一化。具体的操作步骤，已经封装在 CLIPProcessor 里了，你可以不用关心。
        然后，我们再通过CLIPModel，把上面的数值向量，推断成一个表达了图片含义的张量（Tensor）。这里，你就把它当成是一个向量就好了。
    步骤三：同样的，get_text_features也是类似的，先把对应的文本通过CLIPProcessor转换成Token，然后再通过模型推断出表示文本的张量。
    步骤四：我们定义了一个cosine_similarity函数，用来计算张量之间的余弦相似度。
    步骤五：我们利用上面这些函数计算图片和文本之间的相似度。

'''
image_tensor_cat1 = get_image_feature("./data/cat.jpg")
image_tensor_cat2 = get_image_feature("./data/cat1.png")
print("Similarity with cat images : ", cosine_similarity(image_tensor, cat_text_tensor))

#
def get_image_features(image_path):
    # Load the image from the file
    image = Image.open(image_path).convert("RGB")

    with torch.no_grad():
        inputs = processor(images=[image], return_tensors="pt", padding=True)
        inputs.to('cpu')
        features = model.get_image_features(**inputs)
    return features.cpu().numpy()


