import cv2
import os
import numpy as np

def remove_white_border(input_path, output_path):
    # 1. 读取图像并二值化
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 252, 255, cv2.THRESH_BINARY)  # 捕捉高亮白圈

    # 2. 自动检测四边白圈宽度
    def detect_border_width(side_pixels):
        for idx, pixel in enumerate(side_pixels):
            if pixel < 250:  # 遇到非白像素时停止
                return max(0, idx-1)  # 留1像素缓冲
        return len(side_pixels)  # 全白边的情况

    top = detect_border_width(binary[0,:])          # 上边
    bottom = detect_border_width(binary[-1,:])      # 下边
    left = detect_border_width(binary[:,0])         # 左边
    right = detect_border_width(binary[:,-1])       # 右边

    # 3. 精准去除白圈
    cropped = img[top:img.shape[0]-bottom,
                  left:img.shape[1]-right]

    # 4. 保存结果
    cv2.imwrite(output_path, cropped)
    print(f"白圈去除完成，裁剪尺寸：{cropped.shape}")

# 使用示例
input_path = r"F:\My Project\Jupyter\Jupyter\Transfer_Learn_ResNet\Datasets\apots_fliterdata\train\1\0a3202889f4d.png"
output_path =r"F:\My Project\Jupyter\Jupyter\Transfer_Learn_ResNet\Datasets\cai_mix_heibai\vesimg"
output_filename = os.path.basename(input_path).replace('.', '_processed.')  # 生成新文件名
output_path = os.path.join(output_path, output_filename)

remove_white_border(input_path, output_path)

remove_white_border(input_path, output_path)