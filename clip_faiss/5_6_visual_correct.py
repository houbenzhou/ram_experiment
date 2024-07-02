import os
from pathlib import Path

from PIL import Image



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
        return None


def merge_two_images(image_path1, image_path2, direction='horizontal'):
    """
    合成两张图片。
    :param image_path1: 第一张图片的路径
    :param image_path2: 第二张图片的路径
    :param direction: 合成的方向，'horizontal' 或 'vertical'
    :return: 返回合成后的图片对象
    """
    # 打开两张图片
    img1 = Image.open(image_path1)
    img2 = Image.open(image_path2)

    # 根据合成方向计算新图片的尺寸
    if direction == 'horizontal':
        new_width = img1.width + img2.width
        new_height = max(img1.height, img2.height)
    else:  # vertical
        new_width = max(img1.width, img2.width)
        new_height = img1.height + img2.height

    # 创建新的图片对象
    new_img = Image.new('RGB', (new_width, new_height))

    # 将图片粘贴到新的图片对象上
    if direction == 'horizontal':
        new_img.paste(img1, (0, 0))
        new_img.paste(img2, (img1.width, 0))
    else:  # vertical
        new_img.paste(img1, (0, 0))
        new_img.paste(img2, (0, img1.height))

    return new_img

def read_and_parse_file(file_path):
    """
    读取文本文件并解析每一行。
    :param file_path: 文本文件的路径
    """
    line_array=[]
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            line_number = 1
            for line in file:
                # 去除行首行尾的空白字符
                cleaned_line = line.strip()
                line_temp=cleaned_line.split(" ")
                line_array.append(line_temp)
                # 进行一些处理，例如打印行号和内容
                print(f"Line {line_number}: {cleaned_line}")

                # 递增行号
                line_number += 1

    except FileNotFoundError:
        print(f"Error: The file at {file_path} does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

    return line_array


if __name__ == "__main__":

    base_path=os.getcwd()
    faiss_path = os.path.join(base_path, "output", "faiss_model", "35_image_path")
    error_picture_name_logs = os.path.join(faiss_path,"correct_picture_name.txt")
    out_error_picture_path=os.path.join(faiss_path,"correct_picture")
    if not os.path.exists(out_error_picture_path):
        os.makedirs(out_error_picture_path)
    # 使用示例
    line_array=read_and_parse_file(error_picture_name_logs)

    # 替换下面路径为你的图片路径
    for i in line_array:
        image_path1=i[1]
        image_path2=i[3]
        merged_image = merge_two_images(image_path1, image_path2, 'horizontal')

        true_name = extract_directory_name(image_path1, -2)
        pre_name = extract_directory_name(image_path2, -2)
        # 保存图片
        merged_image.save(os.path.join(out_error_picture_path,f"{true_name}_{pre_name}.jpg"))
        # merged_image.save("./sdjhkf.jpg")
        # merged_image.show()  # 显示图
