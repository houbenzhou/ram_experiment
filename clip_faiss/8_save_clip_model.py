import os

from transformers import CLIPProcessor, CLIPModel

proxy_url = "http://192.168.0.30:10809"
os.environ['HTTP_PROXY'] = proxy_url
os.environ['HTTPS_PROXY'] = proxy_url
base_path=os.getcwd()
clip_model_path = os.path.join(base_path,"model" ,"clip_model")

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.save_pretrained(clip_model_path)
processor.save_pretrained(clip_model_path)

