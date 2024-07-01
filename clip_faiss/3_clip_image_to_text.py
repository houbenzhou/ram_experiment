import open_clip
import torch
import os
from PIL import Image
proxy_url="http://192.168.0.30:10809"
os.environ['HTTP_PROXY']=proxy_url
os.environ['HTTPS_PROXY']=proxy_url
model, _, transform =open_clip.create_model_and_transforms('coca_ViT-L-14', pretrained='mscoco_finetuned_laion2B-s13B-b90k')
im=Image.open("./data/cat.jpg").convert("RGB")
im = transform(im).unsqueeze(0)
'''
计算图片的向量，然后基于图片的向量解码出文字

'''
with torch.no_grad(),torch.cuda.amp.autocast():
    genetated=model.generate(im)

print(open_clip.decode(genetated[0]).split("<end_of_text>")[0].replace("<start_of_text>",""))
