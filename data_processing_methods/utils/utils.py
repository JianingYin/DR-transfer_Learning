import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import logging
from datetime import datetime
from skimage.filters import gaussian
from skimage import exposure
import numpy as np
import pandas as pd
from PrePose_Code.utils.storage import draw_confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import trange, tqdm


def logging_config(exp_path='log'):
    # 配置日志

    logger = logging.getLogger()
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(str(exp_path))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def get_subfolder_info(folder_path):
    result = []
    total = 0
    num = 0

    # 遍历文件夹
    for root, dirs, files in os.walk(folder_path):
        # 当前root已经是文件夹的一个层级（包括自身）
        # 如果root不是输入的folder_path（即不是最顶层），则它是子文件夹
        if root != folder_path:
            # 获取当前root相对于folder_path的相对路径（只包含文件夹名）
            rel_path = os.path.relpath(root, folder_path)
            # 获取并存储文件数
            file_count = len(files)
            total += file_count
            num += 1
            result.append([int(rel_path), file_count])

            # 计算每个子文件夹文件数占总文件数的比例，并转化为百分比形式，保留两位小数
    for i in range(num):
        if total > 0:  # 避免除以零的错误
            percentage = round((1 - result[i][1] / total), 2)
            result[i][1] = percentage
        else:
            result[i][1] = 0.0  # 如果total为0，则设置百分比为0

    return result


def find_value_in_2d_list(value, lst):
    for i, row in enumerate(lst):
        if value in row:
            return i  # 返回行索引
    return -1  # 如果没有找到值，返回-1


# 图像处理部分
def image_preprocess(image):
    # 将PIL图像对象转换为NumPy数组
    image = np.array(image)
    # 将图像数据类型转换为float类型，并归一化到[0, 1]的范围内
    image = image.astype(np.float32) / 255.0
    green_channel = image[:, :, 1]  # 提取绿色通道
    # 高斯滤波
    gaussian_filtered = gaussian(green_channel, sigma=1)
    # CLAHE操作
    clahe_processed = exposure.equalize_adapthist(gaussian_filtered, clip_limit=0.03)

    # 将单通道的图像转换为三通道的彩色图像
    processed_image = np.stack([clahe_processed] * 3, axis=-1)
    return processed_image


def collate_fn(batch):
    data = torch.stack([item[0] for item in batch])  # 将图像张量堆叠
    target = torch.tensor([item[1] for item in batch])  # 将标签转换为张量
    return [data, target]


class WeightedCrossEntropyLoss(torch.nn.Module):
    def __init__(self, weight_list, reduction='mean'):
        super(WeightedCrossEntropyLoss, self).__init__()
        # 将二维列表转换为字典
        self.weight_dict = {label: weight for label, weight in weight_list}
        self.reduction = reduction

    def forward(self, inputs, targets):
        # log_probs = F.log_softmax(inputs, dim=1)
        log_probs = F.sigmoid(inputs)

        # 确保所有目标标签都在权重字典中
        # assert all(t in self.weight_dict for t in targets), "Some target labels are not in the weight dict"

        # 根据目标标签从 weight_dict 中获取权重
        weights = torch.tensor([self.weight_dict[t.item()] for t in targets], dtype=torch.float, device=inputs.device)

        # 应用权重到损失
        weighted_loss = -weights * log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # 根据 reduction 参数计算最终的损失
        if self.reduction == 'mean':
            loss = weighted_loss.mean()
        elif self.reduction == 'sum':
            loss = weighted_loss.sum()
        else:
            loss = weighted_loss

        return loss


# 存储
def append_to_csv(filename, data_dict):
    new_row = pd.DataFrame([data_dict])

    if not os.path.isfile(filename):
        new_row.to_csv(filename, index=False)
    else:
        df = pd.read_csv(filename)
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(filename, index=False)


# 保存模型
def save_model(model, path, this_test_acc, best_test_acc):
    weight_path = os.path.join(path, 'weight')
    if not os.path.exists(weight_path):
        os.makedirs(weight_path, exist_ok=True)
    best_name = os.path.join(weight_path, 'best.pt')
    last_name = os.path.join(weight_path, 'last.pt')

    torch.save(model, last_name)

    if os.path.exists(best_name):
        if this_test_acc >= best_test_acc:
            torch.save(model, best_name)
            return this_test_acc, best_name
        else:
            return best_test_acc, best_name
    else:
        torch.save(model, best_name)
        return this_test_acc, best_name


# 训练部分
def train_one_batch(images, labels, device, optimizer, model, criterion):
    images, labels = images.to(device), labels.to(device)
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    _, preds = torch.max(outputs, 1)
    preds, labels = preds.cpu().numpy(), labels.cpu().numpy()

    log_train = {
        'train_loss': loss.item(),
        'train_accuracy': accuracy_score(labels, preds),
        'train_precision': precision_score(labels, preds, average='macro'),
        'train_recall': recall_score(labels, preds, average='macro'),
        'train_f1_score': f1_score(labels, preds, average='macro')
    }
    return log_train


# 测试函数
def test_testset(model, test_loader, device, lable_name=["0", "1", "2", "3", "4"],
                 title="Confusion Matrix of best model", pdf_save_path="Confusion_Matrix.jpg", dpi=300):
    y_gt = []
    y_pred = []
    model.eval()
    loss_list, labels_list, preds_list = [], [], []
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images, labels = images.to(device), labels.to(device)
            labels_pd = model(images)
            predict_np = np.argmax(labels_pd.cpu().detach().numpy(), axis=-1)
            labels_np = labels.cpu().numpy()

            y_pred.append(predict_np)
            y_gt.append(labels_np)
    y_gt_all = np.concatenate(y_gt)
    y_pred_all = np.concatenate(y_pred)
    draw_confusion_matrix(label_true=y_gt_all, label_pred=y_pred_all, label_name=lable_name, title=title,
                          pdf_save_path=pdf_save_path, dpi=dpi)

    log_test = {
        'test_loss': np.mean(loss_list),
        'test_accuracy': accuracy_score(labels_list, preds_list),
        'test_precision': precision_score(labels_list, preds_list, average='macro'),
        'test_recall': recall_score(labels_list, preds_list, average='macro'),
        'test_f1_score': f1_score(labels_list, preds_list, average='macro')
    }

    return log_test


# 验证函数
def evaluate_testset(model, test_loader, device, criterion):
    model.eval()
    loss_list, labels_list, preds_list = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            preds, labels = preds.cpu().numpy(), labels.cpu().numpy()

            loss_list.append(loss.item())
            labels_list.extend(labels)
            preds_list.extend(preds)

    log_test = {
        'test_loss': np.mean(loss_list),
        'test_accuracy': accuracy_score(labels_list, preds_list),
        'test_precision': precision_score(labels_list, preds_list, average='macro'),
        'test_recall': recall_score(labels_list, preds_list, average='macro'),
        'test_f1_score': f1_score(labels_list, preds_list, average='macro')
    }

    return log_test


if __name__ == '__main__':
    dataset_dir = 'all_class'
    # dataset_dir = 'balance_class'
    print(f'数据集路径: {dataset_dir}')
    train_path = os.path.join(dataset_dir, 'train')
    test_path = os.path.join(dataset_dir, 'test')

    TRAIN_RATIO = get_subfolder_info(train_path)
    print(f"train_ratio: {TRAIN_RATIO}")
    TEST_RATIO = get_subfolder_info(test_path)
    print(f"test_ratio: {TEST_RATIO}")

    a = find_value_in_2d_list('2', TRAIN_RATIO)
    print(a)

    criterion = WeightedCrossEntropyLoss(TRAIN_RATIO)
    criterion1 = nn.CrossEntropyLoss()

    # 假设输入和目标与之前相同
    inputs = torch.randn(5, 3, requires_grad=True)
    targets = torch.tensor([1, 0, 2, 1, 0], dtype=torch.long)

    loss = criterion(inputs, targets)
    loss.backward()
    print(loss)

    loss1 = criterion1(inputs, targets)
    loss1.backward()
    print(loss1)

    # 假设output是你的模型输出
    output = torch.tensor([[0.9, 0.7, 0.3, 0.5]])

    # 应用Sigmoid函数
    sigmoid_output = torch.sigmoid(output)

    print(sigmoid_output)