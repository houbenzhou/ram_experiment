import time
import os
from PIL import Image
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



if __name__ == "__main__":
    base_path = r'E:\workspaces\2_temp_projects\classifier'
    input_img_path = os.path.join(base_path, "output")

    # base_path = os.getcwd()
    # input_img_path = os.path.join(base_path, "..","data", "clean_data_5037")

    img_paths=get_image_path(input_img_path)
    total_height=0
    total_weight=0
    for img_path in img_paths:
        image = Image.open(img_path)
        start_time = time.time()
        total_height=total_height+image.height
        total_weight=total_weight+image.width
    average_height=total_height/len(img_paths)
    average_weight=total_weight/len(img_paths)
    print("average_height:",average_height,"average_weight:",average_weight)





