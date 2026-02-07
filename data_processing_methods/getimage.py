#
# import cv2
# import numpy as np
#
# # 输入和输出路径配置
# input_path = r"F:\MyProject\Jupyter\Jupyter\Transfer_Learn_ResNet\Datasets\64c6c6ee0d98_final_gray_invert.png"
# output_dir = r'F:\MyProject\Jupyter\Jupyter\Transfer_Learn_ResNet\Datasets'
#
#
# # 1. 提取绿色通道
# # def extract_green_channel(img_path):
# #     img = cv2.imread(img_path)
# #     if img is None:
# #         raise ValueError("图像加载失败，请检查路径")
# #     # 提取绿色通道（OpenCV中通道顺序为BGR）
# #     green_channel = img[:, :, 1]  # 直接获取单通道灰度图
# #     # 保存绿色通道图像
# #     output_path = f"{output_dir}/64c6c6ee0d98_green.png"
# #     #cv2.imwrite(output_path, green_channel)
# #     return green_channel
#
#
# # 2. 转灰度（若需要，绿色通道已是灰度可跳过）
# # def convert_to_grayscale(img):
# #     # 如果输入是单通道则直接返回
# #     if len(img.shape) == 2:
# #         return img
# #     # 否则转换（这里实际不会执行）
# #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# #     output_path = f"{output_dir}/64c6c6ee0d98_gray.png"
# #     #cv2.imwrite(output_path, gray)
# #     return gray
#
#
#
# # 3. 高斯滤波
# def gaussian_blur(img):
#     blurred = cv2.GaussianBlur(img, (5, 5), 0)  # 核大小5x5，标准差自动计算
#     output_path = f"{output_dir}/64c6c6ee0d98_gaussian1.png"
#     cv2.imwrite(output_path, blurred)
#     return blurred
#
#
# # 4. CLAHE（对比度受限自适应直方图均衡化）
# def apply_clahe(img):
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # 对比度限制2.0，分块8x8
#     clahe_img = clahe.apply(img)
#     output_path = f"{output_dir}/64c6c6ee0d98_clahe1.png"
#     #cv2.imwrite(output_path, clahe_img)
#     return clahe_img
#
#
# # 5. 高斯匹配滤波+归一化
# def gaussian_matched_filter(img):
#     # 高斯滤波（二次滤波增强特征）
#     matched = cv2.GaussianBlur(img, (3, 3), 0)
#     # 归一化到0-255范围
#     normalized = cv2.normalize(matched, None, 0, 255, cv2.NORM_MINMAX)
#     output_path = f"{output_dir}/64c6c6ee0d98_final1.png"
#    # cv2.imwrite(output_path, normalized)
#     return normalized
#
# # def invert_grayscale(img):
# #     """
# #     反转灰度图像：0->255，255->0
# #     使用bitwise_not比255-img更高效
# #     """
# #     inverted = cv2.bitwise_not(img)
# #     output_path = f"{output_dir}/64c6c6ee0d98_final_gray_invert.png"
# #     cv2.imwrite(output_path, inverted)
# #     return inverted
#
#
# # 主流程
# if __name__ == "__main__":
#     # 步骤1：提取绿色通道
#     # green_img = extract_green_channel(input_path)  #
#
#     # 步骤2：转灰度（绿色通道已是单通道，直接传递）
#     # gray_img = convert_to_grayscale(green_img)  #
#
#     # 步骤3：高斯滤波
#     blurred_img = gaussian_blur(input_path)  #
#
#     # 步骤4：CLAHE增强
#     clahe_img = apply_clahe(blurred_img)  #
#
#     # 步骤5：高斯匹配滤波+归一化
#     final_img = gaussian_matched_filter(clahe_img)  #
#
#     # 新增步骤6：灰度翻转
#     #inverted_img = invert_grayscale(final_img)
#
#     print("所有处理完成，反转图像已保存至：", output_dir)
#
#
#
#


import cv2
import os
input_path = r"F:\MyProject\Jupyter\Jupyter\Transfer_Learn_ResNet\Datasets\64c6c6ee0d98_invert_gaussian_clahe.png"
output_dir = r'F:\MyProject\Jupyter\Jupyter\Transfer_Learn_ResNet\Datasets'

def gaussian_matched_filter(img_path):
    """高斯滤波+CLAHE增强复合处理"""
    # 读取图像（强制灰度模式）
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"图像加载失败，路径：{img_path}")

    # 阶段一：高斯滤波去噪
    matched = cv2.GaussianBlur(img, (5, 5), 0)  # 核尺寸与历史操作一致
    normalized = cv2.normalize(matched, None, 0, 255, cv2.NORM_MINMAX)

    # # 阶段二：CLAHE增强
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # 标准参数
    #guiyi_img = normalized.apply(matched)

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)  # 防止路径不存在导致保存失败

    # 规范化的输出路径
    base_name = os.path.basename(img_path).split('.')[0]
    output_path = f"{output_dir}/{base_name}_clahe_guiyi.png"

    # 保存处理结果
    if not cv2.imwrite(output_path, normalized):
        raise IOError(f"文件保存失败，路径：{output_path}")

    return normalized


if __name__ == "__main__":
    try:
        result = gaussian_matched_filter(input_path)
        print(f"处理完成，保存路径：{output_dir}")
    except Exception as e:
        print(f"处理失败：{str(e)}")

        # # 5. 高斯匹配滤波+归一化
        # def gaussian_matched_filter(img):
        #     # 高斯滤波（二次滤波增强特征）
        #     matched = cv2.GaussianBlur(img, (3, 3), 0)
        #     # 归一化到0-255范围
        #     normalized = cv2.normalize(matched, None, 0, 255, cv2.NORM_MINMAX)
        #     output_path = f"{output_dir}/64c6c6ee0d98_final1.png"
        #    # cv2.imwrite(output_path, normalized)
        #     return normalized