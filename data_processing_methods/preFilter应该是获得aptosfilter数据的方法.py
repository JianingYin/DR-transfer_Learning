import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import trange, tqdm
from PrePose_Code.utils.image_processing import CFB
from PrePose_Code.utils.storage import augment_image
# 忽略警告
import warnings

warnings.filterwarnings("ignore")


def get_all_lst(input_dir):
    image_path_lst = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(root, file)
                image_path_lst.append(file_path)
    return image_path_lst


def save_preprocessed_images(input_dir, output_dir):
    image_path_lst = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(root, file)
                image_path_lst.append(file_path)

    for image_path in tqdm(image_path_lst):
        image = CFB(image_path)

        # Determine the relative path for saving
        relative_path = os.path.relpath(image_path, input_dir)

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Save processed image
        output_path = os.path.join(output_dir, relative_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Ensure parent directories exist

        cv2.imwrite(output_path, image)

    print("All images processed and saved.")


if __name__ == "__main__":
    # folder_path = 'F:\\Nico\\weighted_gradient_loss\\datasets\\eye5_balance\\test\\4'
    # image_lst = get_all_lst(folder_path)
    # print(folder_path)

    # for image in tqdm(image_lst):
    #     augment_image(image, n=3)

    input_path = r'F:\My Project\Jupyter\Jupyter\Transfer_Learn_ResNet\split_data'
    output_path = r'F:\My Project\Jupyter\Jupyter\Transfer_Learn_ResNet\Datasets\apots_fliterdata'

    save_preprocessed_images(input_path, output_path)