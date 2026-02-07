import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from PIL import Image


def count_images_in_subcategories(folder_path):
    # 定义存储结果的字典
    results = {}

    # 遍历train和test文件夹
    for main_category in ['train', 'test']:
        main_path = os.path.join(folder_path, main_category)
        if os.path.exists(main_path):
            # 初始化该主类别的字典
            results[main_category] = {}

            # 遍历子类别文件夹
            for sub_category in ['0', '1', '2', '3', '4']:
                sub_path = os.path.join(main_path, sub_category)
                if os.path.exists(sub_path):
                    # 获取子类别中的文件数量
                    image_count = len([f for f in os.listdir(sub_path) if os.path.isfile(os.path.join(sub_path, f))])
                    results[main_category][sub_category] = image_count
                else:
                    results[main_category][sub_category] = 0
        else:
            results[main_category] = {sub_category: 0 for sub_category in ['0', '1', '2', '3', '4']}

    return results


def ensure_subfolder_with_file(target_file_name):
    # 获取当前运行脚本的目录
    current_dir = os.getcwd()

    # 遍历当前目录下的所有子文件夹
    for folder_name in os.listdir(current_dir):
        folder_path = os.path.join(current_dir, folder_name)
        # 检查路径是否为文件夹
        if os.path.isdir(folder_path):
            # 检查该文件夹中是否存在目标文件
            if target_file_name in os.listdir(folder_path):
                return folder_path  # 返回找到的文件夹路径

    # 如果没有找到包含目标文件的文件夹，则创建一个新文件夹
    new_folder_path = os.path.join(current_dir, target_file_name)
    os.makedirs(new_folder_path, exist_ok=True)
    return new_folder_path  # 返回新创建的文件夹路径


def count_subfolders(abs_path):
    # 统计指定路径下的子文件夹数量
    subfolder_count = sum(1 for folder_name in os.listdir(abs_path)
                          if os.path.isdir(os.path.join(abs_path, folder_name)))

    return subfolder_count


def create_folder_addpath(abs_path, folder_name):
    # 构造要创建的文件夹的路径
    folder_path = os.path.join(abs_path, folder_name)

    # 创建文件夹
    os.makedirs(folder_path)

    return folder_path


# 创建exp
def create_exp(root_path='runs', name='train'):
    current_dir = os.getcwd()
    root_lst = os.listdir(current_dir)
    if root_path not in root_lst:
        folder_path = create_folder_addpath(current_dir, root_path)
    else:
        folder_path = os.path.join(current_dir, root_path)
    runs_lst = os.listdir(folder_path)
    if name not in runs_lst:
        created_train = create_folder_addpath(folder_path, name)
    else:
        created_train = os.path.join(folder_path, name)
    subfolder_count = count_subfolders(created_train)
    exp_name = 'exp' + str(subfolder_count)
    created_exp = create_folder_addpath(created_train, exp_name)

    return created_exp


# 绘制混淆矩阵
def draw_confusion_matrix(label_true, label_pred, label_name, title="Confusion Matrix", pdf_save_path=None, dpi=300):
    """

    @param label_true: 真实标签，比如[0,1,2,7,4,5,...]
    @param label_pred: 预测标签，比如[0,5,4,2,1,4,...]
    @param label_name: 标签名字，比如['cat','dog','flower',...]
    @param title: 图标题
    @param pdf_save_path: 是否保存，是则为保存路径pdf_save_path=xxx.png | xxx.pdf | ...等其他plt.savefig支持的保存格式
    @param dpi: 保存到文件的分辨率，论文一般要求至少300dpi
    @return:

    example：
            draw_confusion_matrix(label_true=y_gt,
                          label_pred=y_pred,
                          label_name=["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"],
                          title="Confusion Matrix on Fer2013",
                          pdf_save_path="Confusion_Matrix_on_Fer2013.png",
                          dpi=300)

    """
    cm = confusion_matrix(y_true=label_true, y_pred=label_pred, normalize='true')

    plt.imshow(cm, cmap='Blues')
    plt.title(title)
    plt.xlabel("Predict label")
    plt.ylabel("Truth label")
    plt.yticks(range(label_name.__len__()), label_name)
    plt.xticks(range(label_name.__len__()), label_name, rotation=45)

    plt.tight_layout()

    plt.colorbar()

    for i in range(label_name.__len__()):
        for j in range(label_name.__len__()):
            color = (1, 1, 1) if i == j else (0, 0, 0)  # 对角线字体白色，其他黑色
            value = float(format('%.2f' % cm[j, i]))
            plt.text(i, j, value, verticalalignment='center', horizontalalignment='center', color=color)

    # plt.show()
    if not pdf_save_path is None:
        plt.savefig(pdf_save_path, bbox_inches='tight', dpi=dpi)


def augment_image(image_path, n):
    # 打开原始图片
    original_image = Image.open(image_path)
    # 获取图片的文件名和扩展名
    file_name, file_ext = os.path.splitext(image_path)

    # 计算每次旋转的角度
    angle_step = 360 / n

    # 保存原始图片
    original_image.save(f"{file_name}_0{file_ext}")

    # 旋转并保存图片
    for i in range(1, n):
        rotated_image = original_image.rotate(angle_step * i)
        rotated_image.save(f"{file_name}_{int(angle_step * i)}{file_ext}")


if __name__ == '__main__':
    path = 'F:\\Nico\\weighted_gradient_loss\\datasets\\eye5_balance'
    counts = count_images_in_subcategories(path)
    print(counts)