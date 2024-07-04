import os

def change_file_extension(file_path, new_extension):
    # 分离文件名和旧扩展名
    base_name, _ = os.path.splitext(file_path)
    # 添加新的文件扩展名
    new_file_path = f"{base_name}.{new_extension}"
    os.rename(file_path,new_file_path)
    return new_file_path

def get_file_path(directory):
    # 存储找到的图片路径
    image_paths = []

    # os.walk遍历目录和子目录
    for root, dirs, files in os.walk(directory):
        for file in files:
            # 检查文件是否是图片，这里以几种常见图片格式为例
            if os.path.isfile(os.path.join(root, file)):
                image_paths.append(os.path.join(root, file))

    return image_paths


if __name__ == "__main__":

    base_path = os.getcwd()

    img_path = os.path.join(base_path, "data", "out_clean_data1")
    img_paths = get_file_path(img_path)

    for i in img_paths:

        change_file_extension(i,new_extension='jpg')


