import os
import cv2
import numpy as np
from tqdm import tqdm

def calculate_dataset_stats(root_path):
    """
    计算数据集的均值和标准差 (RGB通道)
    """
    print(f"正在分析路径: {root_path}")
    
    # 递归获取所有图片路径
    img_paths = []
    for subdir, _, files in os.walk(root_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                img_paths.append(os.path.join(subdir, file))

    if not img_paths:
        print("未找到图片文件！")
        return None

    channels_sum, channels_squared_sum, num_pixels = 0, 0, 0

    for path in tqdm(img_paths, desc="计算中"):
        # 读取图片并转为 RGB
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 归一化到 [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # 累加均值和平方和 (用于计算标准差)
        channels_sum += np.mean(img, axis=(0, 1))
        channels_squared_sum += np.mean(img**2, axis=(0, 1))
        num_pixels += 1

    # 计算最终统计量
    mean = channels_sum / num_pixels
    std = np.sqrt(channels_squared_sum / num_pixels - mean**2)

    return mean, std

if __name__ == '__main__':
    # 你的路径
    RAW_PATH = r'F:\DR\Data\aptos\aptos_split'
    FILTER_PATH = r'F:\DR\Data\aptos\aptos_split_filter'

    print("--- 原始彩色图像统计 ---")
    raw_mean, raw_std = calculate_dataset_stats(RAW_PATH)
    print(f"Mean: {raw_mean.tolist()}")
    print(f"Std:  {raw_std.tolist()}")

    print("\n--- 增强血管图像统计 ---")
    filt_mean, filt_std = calculate_dataset_stats(FILTER_PATH)
    print(f"Mean: {filt_mean.tolist()}")
    print(f"Std:  {filt_std.tolist()}")