
import cv2
import numpy as np
import matplotlib.pyplot as plt


def show_image(image, title='Image', cmap_type='gray'):
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap=cmap_type)
    plt.title(title)
    plt.axis('off')
    plt.show()


# Color flipping binarization: 色彩翻转+二值化
def CFB(image_path):
    # read image
    image = cv2.imread(image_path)

    # Grey
    image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Histogram equalization
    # image_equalized = cv2.equalizeHist(image_grey)

    # Create a full 0 matrix
    zero_matrix = np.zeros_like(image_grey)

    # Flipping
    # for i in range(image_grey.shape[0]):
    #     for j in range(image_grey.shape[1]):
    #         zero_matrix[i, j] = 255 - int(image_grey[i, j])
    zero_matrix = 255 - image_grey

    # adaptive_binary_image
    adaptive_binary_image = cv2.adaptiveThreshold(
        zero_matrix,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        199,  # 邻域大小
        -3  # 常数
    )

    return adaptive_binary_image


if __name__ == "__main__":
    # image_path = 'F:\\Nico\\weighted_gradient_loss\\datasets\\balance_class\\test\\3\\99_left_1.44.jpeg'
    image_path = 'F:\\Nico\\weighted_gradient_loss\\datasets\\balance_class\\train\\4\\43839_right_scaled3.jpeg'
    image = CFB(image_path)
    # cv2.imshow('', image)
    # cv2.waitKey(0)
    cv2.imwrite('./grey.jpeg', image)