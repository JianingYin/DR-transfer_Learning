# from PIL import Image
# import numpy as np
# import cv2
#
#
# def process_grayscale_image(input_path, output_path):
#     # 读取灰度图并转换为RGB模式以便修改颜色
#     img = Image.open(input_path).convert('RGB')
#     img_array = np.array(img)
#
#     # 提取灰度图中的白色像素（原灰度图的255对应RGB的(255,255,255)）
#     gray = np.array(Image.open(input_path).convert('L'))  # 确保读取单通道灰度值
#     white_mask = (gray == 255)  # 直接判断灰度值是否为255
#
#     # 生成二值化掩码（白色区域为255，其他为0）
#     binary_mask = np.uint8(white_mask * 255)
#
#     # 连通区域分析（需使用OpenCV）
#     num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
#
#     # 收集各区域面积（排除背景）
#     areas = [(label, stats[label, cv2.CC_STAT_AREA]) for label in range(1, num_labels)]
#     areas.sort(key=lambda x: -x[1])  # 按面积降序排序
#
#     print(areas[:5])
#
#     # 创建输出图像的副本
#     output = img_array.copy()
#
#     # 定义颜色：红(255,0,0)、黄(255,255,0)
#     colors = [(255, 0, 0), (255, 255, 0), (0, 255, 255)]
#
#     # 染色前两大区域
#     for i, (label, _) in enumerate(areas[:3]):
#         if i < len(colors):
#             output[labels == label] = colors[i]
#
#     # 保存结果
#     Image.fromarray(output).save(output_path)
#
# # 示例用法
# if __name__ == '__main__':
#     process_grayscale_image('0a61bddab956.png', '0a61bddab956_change.png')
import os

from PIL import Image
import numpy as np
import cv2


def process_retina_image(input_path, output_path):
    # 读取图像并预处理
    img = Image.open(input_path).convert('RGB')
    img_array = np.array(img)
    h, w = img_array.shape[:2]

    # 提取白色区域
    gray = np.array(Image.open(input_path).convert('L'))
    white_mask = (gray == 255)
    binary_mask = np.uint8(white_mask * 255)

    # 连通域分析获取统计信息
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    # 收集前三区域的边界信息
    top_regions = []
    for label in range(1, num_labels):
        x = stats[label, cv2.CC_STAT_LEFT]
        width = stats[label, cv2.CC_STAT_WIDTH]
        right_x = x + width - 1
        area = stats[label, cv2.CC_STAT_AREA]
        top_regions.append((label, x, right_x, area))

    # 按面积降序取前三
    top_regions.sort(key=lambda x: -x[3])
    top3 = top_regions[:3]

    if not top3:
        return  # 无有效区域

    # 寻找最左和最右区域
    region1 = min(top3, key=lambda x: x[1])  # 最左区域
    region2 = max(top3, key=lambda x: x[2])  # 最右区域

    # 创建彩色副本
    output = img_array.copy()

    # 染色逻辑
    same_region = (region1[0] == region2[0])
    color_red = (0, 0, 0)
    color_yellow = (0, 0, 0)

    # 绘制区域1（红色）
    output[labels == region1[0]] = color_red

    # 绘制区域2（若不同则黄色）
    if not same_region:
        output[labels == region2[0]] = color_yellow

    # 保存结果
    Image.fromarray(output).save(output_path)

def process_retina(input_path):
    # 读取图像并预处理
    img = Image.open(input_path).convert('RGB')
    img_array = np.array(img)
    h, w = img_array.shape[:2]

    # 提取白色区域
    gray = np.array(Image.open(input_path).convert('L'))
    white_mask = (gray == 255)
    binary_mask = np.uint8(white_mask * 255)

    # 连通域分析获取统计信息
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    # 收集前三区域的边界信息
    top_regions = []
    for label in range(1, num_labels):
        x = stats[label, cv2.CC_STAT_LEFT]
        width = stats[label, cv2.CC_STAT_WIDTH]
        right_x = x + width - 1
        area = stats[label, cv2.CC_STAT_AREA]
        top_regions.append((label, x, right_x, area))

    # 按面积降序取前三
    top_regions.sort(key=lambda x: -x[3])
    top3 = top_regions[:3]

    if not top3:
        return  # 无有效区域

    # 寻找最左和最右区域
    region1 = min(top3, key=lambda x: x[1])  # 最左区域
    region2 = max(top3, key=lambda x: x[2])  # 最右区域

    # 创建彩色副本
    output = img_array.copy()

    # 染色逻辑
    same_region = (region1[0] == region2[0])
    color_red = (0, 0, 0)
    color_yellow = (0, 0, 0)

    # 绘制区域1（红色）
    output[labels == region1[0]] = color_red

    # 绘制区域2（若不同则黄色）
    if not same_region:
        output[labels == region2[0]] = color_yellow

    # 保存结果
    return output, os.path.basename(input_path)


if __name__ == '__main__':
    # fold_dir = os.listdir('./')
    # png_files = [file for file in fold_dir if file.endswith('.png')]
    # print(png_files)
    # # 使用示例
    # for png_file in png_files:
    #     new_filename = os.path.splitext(png_file)[0] + "_black.png"
    #     process_retina_image(png_file, new_filename)

    output, basename = process_retina(r"F:\My Project\Jupyter\Jupyter\Transfer_Learn_ResNet\Datasets\apots_fliterdata\train\1\0a3202889f4d.png")
    print(type(output))
    print((output.shape))
    print(basename)