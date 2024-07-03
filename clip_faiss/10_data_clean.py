import os
from pathlib import Path

from PIL import Image
import shutil
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


import os


def check_files_in_directory(directory_path):
    # 检查目录路径是否存在
    if not os.path.exists(directory_path):
        return "指定的目录不存在。"

    # os.listdir列出目录中的所有文件和文件夹
    files = os.listdir(directory_path)

    # # 筛选出文件（排除文件夹）
    # files = [file for file in files if os.path.isfile(os.path.join(directory_path, file))]
    #
    # if files:
    #     return f"目录中有文件。具体文件数: {len(files)}"
    # else:
    #     return "目录中没有文件。"
    return files

def copy_folder(src, dst):
    # 确保目标目录存在，如果不存在则创建
    if not os.path.exists(dst):
        os.makedirs(dst)
    # 使用shutil.copytree来拷贝目录
    # 如果目标目录已存在，我们需要使用shutil.copy来逐个复制文件
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)  # Python 3.8+ 支持dirs_exist_ok参数
        else:
            shutil.copy2(s, d)


if __name__ == "__main__":
    base_path = os.getcwd()

    img_path = os.path.join(base_path, "data", "Trax_bbox出来的小图含label_20230207")
    img_paths = get_image_path(img_path)
    img_path_10 = os.path.join(base_path, "data", "Trax_bbox出来的小图含label_20230207_cropped_images_10")
    img_paths_10 = get_image_path(img_path)
    out_5000_clean_data = os.path.join(base_path, "data", "out_clean_data")
    category_names_10 = []

    # 你可以替换这里的路径来使用这个函数
    # directory_path = '输入你的文件夹路径'
    img_name=check_files_in_directory(img_path)
    img_10_name = check_files_in_directory(img_path_10)
    no_image_data_name=[]
    for i in img_name:
        if i not in img_10_name:
            no_image_data_name.append(i)

    for i in img_name:
        if i not in no_image_data_name:
            copy_folder(os.path.join(img_path,i),os.path.join(out_5000_clean_data,i))



    # # 使用示例
    # for i in img_paths_10:
    #
    #     category_name=extract_directory_name(i,-2)
    #     category_names_10.append(category_name)
    #
    # for i in img_paths:
    #     category_name = extract_directory_name(i, -2)
    #     if category_name in category_names_10:
    #         continue
    #     else:
    #         print(i)



