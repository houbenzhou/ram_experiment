import time

import matplotlib.pyplot as plt
import pandas as pd


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
def calculate_image_mean(img_paths):
    # 打开图像

    for img_path in img_paths:
        img = Image.open(img_path)
        # 将图像数据转换为numpy数组
        img_array = np.array(img)
        mean_value = img_array.mean()
    # 如果图像是彩色的，img_array将是一个三维数组 (高度, 宽度, 颜色通道)
    # 计算所有像素值的平均数

    print("average_height:", average_height, "average_weight:", average_weight)
    return mean_value



def count_image_h_w(img_paths):
    total_height=0
    total_weight=0
    for img_path in img_paths:
        image = Image.open(img_path)
        start_time = time.time()
        total_height=total_height+image.height
        total_weight=total_weight+image.width
    average_height=total_height/len(img_paths)
    average_weight=total_weight/len(img_paths)
    return average_height,average_weight


from PIL import Image
import numpy as np
import os


def read_image_pixels(image_path):
    """读取单个图像的像素数据"""
    # 打开图像
    img = Image.open(image_path)
    # 将图像转换为numpy数组并归一化到0-1范围
    img_array = np.array(img, dtype=np.float32) / 255.0
    return img_array


def calculate_overall_stats(image_path):
    """计算所有图像的整体均值和方差"""
    all_pixels = []
    # 遍历文件夹中的所有文件
    for img_path in img_paths:
        img_pixels = read_image_pixels(img_path)
        all_pixels.append(img_pixels)
    # 合并所有图像的像素数据
    all_pixels_array = np.concatenate([pixels.ravel() for pixels in all_pixels])

    # 计算整体均值和方差
    overall_mean = all_pixels_array.mean()
    overall_variance = all_pixels_array.var()

    return overall_mean, overall_variance
def get_image_dimensions(image_path):
    """获取单个图像的尺寸"""
    with Image.open(image_path) as img:
        return img.width, img.height


def dimensions_distribution(img_paths):
    """统计文件夹中所有图像的尺寸分布"""
    dimensions = []
    # 遍历文件夹中的所有文件

    for img_path in img_paths:
        width, height = get_image_dimensions(img_path)
        dimensions.append((width, height))
    # 将尺寸数据转换为DataFrame
    return pd.DataFrame(dimensions, columns=['Width', 'Height'])


def plot_histograms(df):
    """绘制宽度和高度的直方图"""
    plt.figure(figsize=(12, 5))

    # 绘制宽度的直方图
    plt.subplot(1, 2, 1)
    plt.hist(df['Width'], bins=30, color='blue', edgecolor='black')
    plt.title('Width Distribution')
    plt.xlabel('Width')
    plt.ylabel('Frequency')

    # 绘制高度的直方图
    plt.subplot(1, 2, 2)
    plt.hist(df['Height'], bins=30, color='green', edgecolor='black')
    plt.title('Height Distribution')
    plt.xlabel('Height')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    base_path = r'E:\workspaces\2_temp_projects\classifier'
    # input_img_path = os.path.join(base_path, "output")
    input_img_path = r"E:\workspaces\1_my_projects\ram_experiment\clip_faiss\data\新零售图片数据_Trax_部分"
    img_paths=get_image_path(input_img_path)
    average_height,average_weight=count_image_h_w(img_paths)
    print("average_height:",average_height,"average_weight:",average_weight)
    # 替换 'your_image_folder' 为你的图像文件夹路径
    overall_mean, overall_variance = calculate_overall_stats(img_paths)
    print(f"Overall Mean: {overall_mean}, Overall Variance: {overall_variance}")
    df_dimensions = dimensions_distribution(img_paths)
    plot_histograms(df_dimensions)




