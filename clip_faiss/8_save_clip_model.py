import os

from transformers import CLIPProcessor, CLIPModel, AutoImageProcessor, AutoModel

proxy_url = "http://192.168.0.148:10809"
os.environ['HTTP_PROXY'] = proxy_url
os.environ['HTTPS_PROXY'] = proxy_url
base_path=os.getcwd()
# clip model
# clip_model_path = os.path.join(base_path,"model" ,"clip_model")
#
# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
# model.save_pretrained(clip_model_path)
# processor.save_pretrained(clip_model_path)
# dinov2 model
dinov2_model_path = os.path.join(base_path,"model" ,"dinov2_model")

dinov2_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
dinov2_model = AutoModel.from_pretrained("facebook/dinov2-base")
dinov2_model.save_pretrained(dinov2_model_path)
dinov2_processor.save_pretrained(dinov2_model_path)

