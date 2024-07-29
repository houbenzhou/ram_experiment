import os
from pathlib import Path

from PIL import Image

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

    return image_paths
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
        print(path)
        return None
def crop_image(image_path, output_path,crop_rate=0.1):
    # 打开图片
    img = Image.open(image_path)

    # 获取图片的宽度和高度
    width, height = img.size

    # 计算裁切的大小：10%的宽和高
    left = width * crop_rate
    top = height * crop_rate
    right = width * (1-crop_rate)
    bottom = height * (1-crop_rate)

    # 根据计算的边界裁切图片
    cropped_img = img.crop((left, top, right, bottom))

    # 保存裁切后的图片
    cropped_img.save(output_path)
    # print(f'Cropped image saved to {output_path}')


# 使用示例


if __name__ == "__main__":
    base_path = os.getcwd()

    # img_path = os.path.join(base_path, "data", "新零售图片数据_Trax_部分")
    # img_paths = get_image_path(img_path)
    # 输出35类按照0.1的比例切除图像
    # out_path = os.path.join(base_path, "data", "新零售图片数据_Trax_部分_cropped_images_10")
    # crop_rate=0.1
    # 输出35类按照0.2的比例切除图像
    # out_path = os.path.join(base_path, "data", "新零售图片数据_Trax_部分_cropped_images_20")
    # crop_rate=0.2

    img_path = os.path.join(base_path,"..", "data", "out_clean_data")
    img_paths = get_image_path(img_path)
    # # 输出5037类按照0.1的比例切除图像
    # out_path = os.path.join(base_path, "data", "out_clean_data_cropped_images_10")
    # crop_rate=0.1
    # # 输出5037类按照0.2的比例切除图像
    out_path = os.path.join(base_path, "..","data", "out_clean_data_cropped_images_20")
    crop_rate=0.2
    # 如果out_path不存在则创建
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    # 使用示例
    for i in img_paths:
        file_name=os.path.basename(i)
        category_name=extract_directory_name(i,-2)
        out_sub_path=os.path.join(out_path,category_name)
        if not os.path.exists(out_sub_path):
            os.makedirs(out_sub_path)
        crop_image(i, os.path.join(out_sub_path,file_name),crop_rate=crop_rate)
