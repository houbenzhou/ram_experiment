import os
from PIL import Image

from PIL import Image

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
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            line_number = 1
            for line in file:
                # 去除行首行尾的空白字符
                cleaned_line = line.strip()
                line_array=cleaned_line.split(" ")
                # 进行一些处理，例如打印行号和内容
                print(f"Line {line_number}: {cleaned_line}")

                # 递增行号
                line_number += 1

    except FileNotFoundError:
        print(f"Error: The file at {file_path} does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":

    base_images=os.getcwd()
    img_path = os.path.join(base_images,"data","新零售图片数据_Trax_部分")
    file_name = os.path.join(base_images,"output","新零售图片数据_Trax_部分.txt")

    # 使用示例
    # 替换下面路径为你的文本文件路径
    read_and_parse_file(file_name)

    # 替换下面路径为你的图片路径
    merged_image = merge_two_images('path/to/first/image.jpg', 'path/to/second/image.jpg', 'horizontal')
    merged_image.show()  # 显示图
