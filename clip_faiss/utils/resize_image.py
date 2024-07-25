from PIL import Image
import os
from pathlib import Path
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


from PIL import Image


def resize_image_by_scale(input_path, output_path, scale):
    # 打开图像文件
    with Image.open(input_path) as img:
        original_width, original_height = img.size

        # 计算新的尺寸
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)

        # 调整图像尺寸
        resized_img = img.resize((new_width, new_height))
        # 保存调整后的图像
        resized_img.save(output_path)
        print("Image resized successfully.")
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



if __name__ == "__main__":
    # base_path = r'E:\workspaces\2_temp_projects\classifier'
    # input_img_path = os.path.join(base_path, "output")

    base_path = os.getcwd()
    input_img_path = os.path.join(base_path, "..","data", "clean_data_5037")
    output_path = os.path.join(base_path, "..","data", "clean_data_5037_resize")

    scale = 0.5  # 缩放比例，0.5意味着缩小到原始尺寸的一半

    img_paths=get_image_path(input_img_path)

    for img_path in img_paths:
        file_name=os.path.basename(img_path)
        category_name = extract_directory_name(img_path, -2)
        out_sub_path = os.path.join(output_path, category_name)
        if not os.path.exists(out_sub_path):
            os.makedirs(out_sub_path)
        output_path_file = os.path.join(out_sub_path, file_name)
        resize_image_by_scale(img_path, output_path_file, scale)


