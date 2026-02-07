import cv2
import numpy as np
import os

from del_white_去周围白圈 import process_retina

from tqdm import trange, tqdm

def process_image(input_path, output_dir):
    process_retina_out, basename = process_retina(input_path)
    # 1. 读取图像并转为灰度
    img = cv2.cvtColor(process_retina_out, cv2.COLOR_RGB2GRAY)
    # img = input_path
    h, w = img.shape

    # 2. 动态划分区域（添加重叠区域）
    rows = 3 if h < 400 else 4
    cols = 4 if w < 600 else 3
    overlap = 5  # 边界扩展像素数
    # print(f"划分区域：{rows}行×{cols}列，共{rows * cols}个区域")

    # 3. 创建空画布存储处理结果
    result = np.zeros_like(img)

    # 4. 遍历每个区域进行处理（带边界扩展）
    for i in range(rows):
        for j in range(cols):
            # 计算扩展后的区域边界
            y_start = max(0, i * h // rows - overlap)
            y_end = min(h, (i + 1) * h // rows + overlap) if i != rows - 1 else h
            x_start = max(0, j * w // cols - overlap)
            x_end = min(w, (j + 1) * w // cols + overlap) if j != cols - 1 else w

            # 提取扩展后的子区域
            region = img[y_start:y_end, x_start:x_end]

            # 筛选连通区域
            _, labels = cv2.connectedComponents(region, connectivity=8)
            unique_labels = np.unique(labels)

            for label in unique_labels:
                if label == 0: continue

                # 生成当前连通区域的掩膜
                mask = (labels == label).astype(np.uint8) * 255

                # 计算面积和像素范围
                area = cv2.countNonZero(mask)
                current_pixels = region[labels == label]
                pixel_min, pixel_max = current_pixels.min(), current_pixels.max()



                # 条件筛选
                if area >= 200 and (pixel_max - pixel_min) <= 31 :
                    # 将掩膜合并到原始区域（去除扩展部分）
                    orig_y_start = i * h // rows
                    orig_y_end = (i + 1) * h // rows if i != rows - 1 else h
                    orig_x_start = j * w // cols
                    orig_x_end = (j + 1) * w // cols if j != cols - 1 else w

                    # 计算掩膜在原始区域的偏移
                    mask_cropped = mask[
                                   (orig_y_start - y_start):(orig_y_end - y_start),
                                   (orig_x_start - x_start):(orig_x_end - x_start)
                                   ]

                    # 按位或合并到结果
                    result[orig_y_start:orig_y_end, orig_x_start:orig_x_end] = \
                        cv2.bitwise_or(result[orig_y_start:orig_y_end, orig_x_start:orig_x_end], mask_cropped)


    # 5. 全局连通域合并
    # 形态学闭运算连接边缘
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)

    # 二次连通域分析，合并跨区域目标
    _, labels = cv2.connectedComponents(result, connectivity=8)
    unique_labels = np.unique(labels)

    final_result = np.zeros_like(result)
    for label in unique_labels:
        if label == 0: continue
        mask = (labels == label).astype(np.uint8) * 255
        final_result = cv2.bitwise_or(final_result, mask)

    # 6. 保存结果
    output_path = os.path.join(output_dir, basename)
    cv2.imwrite(output_path, final_result)
    # print(f"处理完成！结果保存至：{output_path}")
# # 执行处理
# input_path = r"F:\My Project\Jupyter\Jupyter\Transfer_Learn_ResNet\Datasets\apots_fliterdata\train\2\00e4ddff966a.png"
# output_dir = r"F:\My Project\Jupyter\Jupyter\Transfer_Learn_ResNet\Datasets\cai_mix_heibai\vesimg"
#
# process_image(input_path, output_dir)

# 需要处理的文件夹-目标文件夹

my_lst = [
    [r"F:\My Project\Jupyter\Jupyter\Transfer_Learn_ResNet\Datasets\apots_fliterdata\test\0", r"F:\My Project\Jupyter\Jupyter\Transfer_Learn_ResNet\Datasets\aptos_ves\test\0"],
    [r"F:\My Project\Jupyter\Jupyter\Transfer_Learn_ResNet\Datasets\apots_fliterdata\test\1", r"F:\My Project\Jupyter\Jupyter\Transfer_Learn_ResNet\Datasets\aptos_ves\test\1"],
    [r"F:\My Project\Jupyter\Jupyter\Transfer_Learn_ResNet\Datasets\apots_fliterdata\test\2", r"F:\My Project\Jupyter\Jupyter\Transfer_Learn_ResNet\Datasets\aptos_ves\test\2"],
    [r"F:\My Project\Jupyter\Jupyter\Transfer_Learn_ResNet\Datasets\apots_fliterdata\test\3", r"F:\My Project\Jupyter\Jupyter\Transfer_Learn_ResNet\Datasets\aptos_ves\test\3"],
    [r"F:\My Project\Jupyter\Jupyter\Transfer_Learn_ResNet\Datasets\apots_fliterdata\train\0", r"F:\My Project\Jupyter\Jupyter\Transfer_Learn_ResNet\Datasets\aptos_ves\train\0"],
    [r"F:\My Project\Jupyter\Jupyter\Transfer_Learn_ResNet\Datasets\apots_fliterdata\train\1", r"F:\My Project\Jupyter\Jupyter\Transfer_Learn_ResNet\Datasets\aptos_ves\train\1"],
    [r"F:\My Project\Jupyter\Jupyter\Transfer_Learn_ResNet\Datasets\apots_fliterdata\train\2", r"F:\My Project\Jupyter\Jupyter\Transfer_Learn_ResNet\Datasets\aptos_ves\train\2"],
    [r"F:\My Project\Jupyter\Jupyter\Transfer_Learn_ResNet\Datasets\apots_fliterdata\train\3", r"F:\My Project\Jupyter\Jupyter\Transfer_Learn_ResNet\Datasets\aptos_ves\train\3"],
    [r"F:\My Project\Jupyter\Jupyter\Transfer_Learn_ResNet\Datasets\apots_fliterdata\train\4", r"F:\My Project\Jupyter\Jupyter\Transfer_Learn_ResNet\Datasets\aptos_ves\train\4"],
]

for i in range(len(my_lst)):
    folder = my_lst[i][0]
    output_dir = my_lst[i][1]
    print("*"*80)
    print(f"输入路径： {folder}")
    print(f"输出路径： {output_dir}")
    print("*" * 80)

    listdir = os.listdir(folder)
    for file in tqdm(listdir):
        input_path = os.path.join(folder, str(file))
        process_image(input_path, output_dir)


##### folder为输入的路径，output_dir为输出路径，写文件夹名 ###
# folder = r"F:\My Project\Jupyter\Jupyter\Transfer_Learn_ResNet\Datasets\apots_fliterdata\test\4"
# output_dir = r"F:\My Project\Jupyter\Jupyter\Transfer_Learn_ResNet\Datasets\aptos_ves\test\4"
#
# listdir = os.listdir(folder)
# for file in listdir:
#     input_path = os.path.join(folder, str(file))
#     process_image(input_path, output_dir)

##########################################################

# ########  用面积和相邻像素值的大小来划定区域
#
# import cv2
# import numpy as np
# import os
#
#
# def process_image(input_path, output_dir):
#     # 1. 读取图像并转为灰度
#     img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
#     h, w = img.shape
#
#     # 2. 动态划分区域
#     rows = 3 if h < 400 else 4
#     cols = 4 if w < 600 else 3
#     print(f"划分区域：{rows}行×{cols}列，共{rows * cols}个区域")
#
#     # 3. 创建空画布存储处理结果
#     result = np.zeros_like(img)
#
#     # 4. 遍历每个区域进行处理
#     for i in range(rows):
#         for j in range(cols):
#             # 计算区域边界
#             y_start = i * h // rows
#             y_end = (i + 1) * h // rows if i != rows - 1 else h
#             x_start = j * w // cols
#             x_end = (j + 1) * w // cols if j != cols - 1 else w
#
#             # 提取子区域
#             region = img[y_start:y_end, x_start:x_end]
#
#             # 筛选连通区域
#             _, labels = cv2.connectedComponents(region, connectivity=8)
#             unique_labels = np.unique(labels)
#
#             for label in unique_labels:
#                 if label == 0: continue  # 跳过背景
#
#                 # 生成当前连通区域的掩膜
#                 mask = (labels == label).astype(np.uint8) * 255
#
#                 # 计算相邻像素数量（即连通区域面积）
#                 area = cv2.countNonZero(mask)
#
#                 # 获取当前区域像素值范围
#                 current_pixels = region[labels == label]
#                 pixel_min = current_pixels.min()
#                 pixel_max = current_pixels.max()
#
#                 # 新条件：面积≥10且像素变化≤3
#                 if area >= 500 and (pixel_max - pixel_min) <= 15:
#                     result[y_start:y_end, x_start:x_end] = cv2.bitwise_or(
#                         result[y_start:y_end, x_start:x_end], mask)
#
#     # 5. 保存结果
#     output_path = os.path.join(output_dir, os.path.basename(input_path))
#     cv2.imwrite(output_path, result)
#     print(f"处理完成！结果保存至：{output_path}")
#
#
# # 执行处理
# input_path = r"F:\My Project\Jupyter\Jupyter\Transfer_Learn_ResNet\Datasets\cai_mix_heibai\train\1\mix_0a3202889f4d.png"
# output_dir = r"F:\My Project\Jupyter\Jupyter\Transfer_Learn_ResNet\Datasets\cai_mix_heibai\vesimg"
# process_image(input_path, output_dir)


## 查像素值的分布范围
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import os
#
# # 输入输出路径
# input_path = r"F:\My Project\Jupyter\Jupyter\Transfer_Learn_ResNet\Datasets\cai_mix_heibai\train\2\mix_3fd7df6099e3.png"
# output_dir = r"F:\My Project\Jupyter\Jupyter\Transfer_Learn_ResNet\Datasets\cai_mix_heibai\vesimg"
#
# # 读取图像并检查
# img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)  # 按灰度图读取
# if img is None:
#     raise FileNotFoundError(f"无法读取图像，请检查路径：{input_path}")
#
# # 计算直方图（参考网页1、网页62）
# hist = cv2.calcHist([img], [0], None, [256], [0, 256])
#
# # 计算像素统计值（参考网页34、网页35）
# min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(img)
# mean_val, stddev_val = cv2.meanStdDev(img)
#
# # 创建可视化图表
# plt.figure(figsize=(12, 6))
#
# # 直方图绘制
# plt.subplot(121)
# plt.plot(hist, color='black')
# plt.title('Pixel Value Histogram')
# plt.xlabel('Pixel Intensity (0-255)')
# plt.ylabel('Frequency')
# plt.grid(linestyle='--')
#
# # 统计信息标注
# text = f"Min: {min_val:.1f}\nMax: {max_val:.1f}\nMean: {mean_val[0][0]:.1f}\nStdDev: {stddev_val[0][0]:.1f}"
# plt.subplot(122)
# plt.text(0.1, 0.5, text, fontsize=12, va='center')
# plt.axis('off')
#
# # 保存结果
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)
# output_path = os.path.join(output_dir, "pixel_3fd7df6099e3.png")
# plt.savefig(output_path, dpi=300, bbox_inches='tight')
# plt.close()  # 关闭图表避免内存泄漏
#
# print(f"分析完成！结果已保存至：{output_path}")

