import os
from pypinyin import pinyin, Style
def list_subdirectories(path='.'):
    """列出指定路径下的所有子目录"""
    subdirectories = []
    with os.scandir(path) as entries:
        for entry in entries:
            if entry.is_dir():
                subdirectories.append(entry.name)
    return subdirectories

base_path = os.getcwd()
current_path = os.path.join(base_path, '..', 'data', 'clean_data_5037')
out_file = os.path.join(base_path, 'category_name_5036.txt')
# current_path = os.path.join(base_path, '..', 'data', 'clean_data_5037_correct_3_2times_3')
# out_file = os.path.join(base_path, 'category_name_3058.txt')
subdirectories = list_subdirectories(current_path)
subdirectories.sort(key=lambda x: ''.join([a[0] for a in pinyin(x, style=Style.TONE3)]))

# # 使用with语句打开文件，确保正确地关闭文件
# with open(out_file, 'w', encoding='utf-8') as file:
#     for directory in subdirectories:
#         # 写入每个目录名，后面加上换行符
#         file.write(directory + '\n')
# 使用with语句打开文件，确保正确地关闭文件
with open(out_file, 'w', encoding='utf-8') as file:
    # 使用zip和iter创建两列的效果
    args = [iter(subdirectories)] * 2
    for a, b in zip(*args):
        # 格式化输出，对齐两列
        file.write(f'{a:<10}    {b:<10}\n')

