
###############################                                                       ###############################
#                              将彩色图与黑白图直接进行逐点相乘，当黑白图为0时进行相加非零时相乘
###############################                                                       ###############################


'''import cv2
import numpy as np
import os


def process_images(color_path, vessel_path, output_path):
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)

    # 遍历彩色图像目录
    for filename in os.listdir(color_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # 读取彩色图像
            color_img = cv2.imread(os.path.join(color_path, filename))

            # 分离通道并处理绿色通道
            b, g, r = cv2.split(color_img)
            denoised_g = cv2.GaussianBlur(g, (0, 0), sigmaX=1)  # 自动计算核大小

            # 读取对应的血管图（假设文件名相同）
            vessel_img = cv2.imread(os.path.join(vessel_path, filename), cv2.IMREAD_GRAYSCALE)

            if vessel_img is None:
                print(f"警告：未找到对应的血管图 {filename}")
                continue

            # 血管图去噪
            denoised_vessel = cv2.GaussianBlur(vessel_img, (0, 0), sigmaX=1)

            # 创建三通道绿色图像
            green_3ch = np.stack([denoised_g] * 3, axis=-1)

            # 条件运算预处理
            vessel_mask = denoised_vessel.astype(np.float32) / 255.0
            green_float = denoised_g.astype(np.float32) / 255.0

            # 执行条件运算
            result = np.zeros_like(green_3ch, dtype=np.float32)
            for c in range(3):
                # 当血管像素为0时相加，非零时相乘
                result[:, :, c] = np.where(vessel_mask == 0,
                                           green_float + vessel_mask,
                                           green_float * vessel_mask)

            # 转换回uint8并保存
            result = np.clip(result * 255, 0, 255).astype(np.uint8)
            output_filename = os.path.join(output_path, filename)
            cv2.imwrite(output_filename, result)
            print(f"已处理：{filename}")


if __name__ == "__main__":
    color_dir = r"F:\MyProject\Jupyter\Jupyter\Transfer_Learn_ResNet\split_data\train\0"
    vessel_dir = r"F:\MyProject\Jupyter\Jupyter\Transfer_Learn_ResNet\Datasets\aptos_ves\train\0"
    output_dir = r"F:\MyProject\Jupyter\Jupyter\Transfer_Learn_ResNet\Datasets\greenvesplus\g8plusv2\train\0"

    process_images(color_dir, vessel_dir, output_dir)'''


############################                                                                      ###############################
#                              （彩色图*0.8）*（黑白图*0.2）后再进行逐点相乘，当黑白图为0时进行相加非零时相乘
############################                                                                      ###############################


import cv2
import numpy as np
import os


def process_images(color_path, vessel_path, output_path):
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)

    # 遍历彩色图像目录
    for filename in os.listdir(color_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # 读取彩色图像
            color_img = cv2.imread(os.path.join(color_path, filename))

            # 分离通道并处理绿色通道
            b, g, r = cv2.split(color_img)
            denoised_g = cv2.GaussianBlur(g, (0, 0), sigmaX=1)  # 自动计算核大小

            # 应用0.8权重到绿色通道
            weighted_g = (denoised_g * 1.5).astype(np.uint8)

            # 读取对应的血管图（假设文件名相同）
            vessel_img = cv2.imread(os.path.join(vessel_path, filename), cv2.IMREAD_GRAYSCALE)

            if vessel_img is None:
                print(f"警告：未找到对应的血管图 {filename}")
                continue

            # 血管图去噪并应用0.2权重
            denoised_vessel = cv2.GaussianBlur(vessel_img, (0, 0), sigmaX=1)
            weighted_vessel = (denoised_vessel * 1).astype(np.uint8)

            # 创建三通道加权绿色图像
            weighted_green_3ch = np.stack([weighted_g] * 3, axis=-1)

            # 预处理为浮点类型
            vessel_mask = weighted_vessel.astype(np.float32) / 255.0
            green_float = weighted_g.astype(np.float32) / 255.0

            # 执行条件运算
            result = np.zeros_like(weighted_green_3ch, dtype=np.float32)
            for c in range(3):
                # 当血管像素为0时相加，非零时相乘
                result[:, :, c] = np.where(vessel_mask == 0,
                                           green_float + vessel_mask,
                                           green_float * vessel_mask)

            # 转换回uint8并保存
            result = np.clip(result * 255, 0, 255).astype(np.uint8)
            output_filename = os.path.join(output_path, filename)
            cv2.imwrite(output_filename, result)
            print(f"已处理：{filename}")


if __name__ == "__main__":
    color_dir = r"F:\MyProject\Jupyter\Jupyter\Transfer_Learn_ResNet\split_data\train\4"
    vessel_dir = r"F:\MyProject\Jupyter\Jupyter\Transfer_Learn_ResNet\Datasets\aptos_ves\train\4"
    output_dir = r"F:\MyProject\Jupyter\Jupyter\Transfer_Learn_ResNet\Datasets\greenvesplus\g15plusv10\train\4"

    process_images(color_dir, vessel_dir, output_dir)






























