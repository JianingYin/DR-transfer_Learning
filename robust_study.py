# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, roc_curve, auc)
from sklearn.preprocessing import label_binarize
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from efficientnet_pytorch import EfficientNet
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True # 防止个别 PNG 损坏导致训练中断
import cv2

import random

def apply_random_mask(img_tensor, mask_ratio=0.3):
    """
    在 Tensor 图像上随机生成黑色矩形，直到遮挡面积达到 mask_ratio
    img_tensor: [C, H, W]
    """
    c, h, w = img_tensor.shape
    mask_img = img_tensor.clone()
    total_pixels = h * w
    masked_pixels = 0
    
    # 循环添加遮挡块，直到达到目标比例
    while masked_pixels / total_pixels < mask_ratio:
        # 随机生成矩形尺寸（图像尺寸的 5% 到 20% 之间）
        rh = random.randint(int(h*0.05), int(h*0.2))
        rw = random.randint(int(w*0.05), int(w*0.2))
        
        # 随机位置
        rx = random.randint(0, h - rh)
        ry = random.randint(0, w - rw)
        
        # 遮掩为黑色 (注意：Normalize 后的 0 像素值可能不是 0，但通常设为最小值或均值处)
        # 这里直接设为 Normalize 后的“零点”
        mask_img[:, rx:rx+rh, ry:ry+rw] = 0 
        
        masked_pixels += rh * rw
        
    return mask_img



# ================= 1. 数据对齐 Dataset =================
class DualInputDataset(Dataset):
    def __init__(self, raw_dir, filter_dir, raw_transform=None, filt_transform=None, noise_sigma=0.0, apply_mask=False, apply_erode=False, apply_zero=False): # 默认不加
        self.raw_dataset = datasets.ImageFolder(raw_dir)
        self.filter_dir = filter_dir
        self.raw_transform = raw_transform
        self.filt_transform = filt_transform
        self.classes = self.raw_dataset.classes
        # 新增：噪声强度参数 (0.0 表示不加噪声)
        self.noise_sigma = noise_sigma
        self.apply_mask = apply_mask # 记录是否开启遮掩 
        self.apply_erode = apply_erode
        self.apply_zero = apply_zero # 记录是否开启完全置零

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, index):
            raw_img_path, label = self.raw_dataset.samples[index]
            rel_path = os.path.relpath(raw_img_path, self.raw_dataset.root)
            filter_img_path = os.path.join(self.filter_dir, rel_path)
            
            raw_img = Image.open(raw_img_path).convert('RGB')
            filter_img = Image.open(filter_img_path).convert('RGB')

    # --- 实验 C: 形态学腐蚀 (在 Transform 之前处理 PIL 对象) ---
            if self.apply_erode:
                # 1. 转为 Numpy
                img_np = np.array(filter_img)
                # 2. 定义 5x5 卷积核
                kernel = np.ones((5, 5), np.uint8)
                # 3. 执行腐蚀 (模拟细小血管变细、消失)
                img_np = cv2.erode(img_np, kernel, iterations=1)
                # 4. 转回 PIL
                filter_img = Image.fromarray(img_np)

            # --- 执行正常的 Transform (Resize, ToTensor, Normalize) ---
            if self.raw_transform:
                raw_img = self.raw_transform(raw_img)
            if self.filt_transform:
                filter_img = self.filt_transform(filter_img)

            # --- 实验 D: 完全缺失 ---
            if self.apply_zero: # 新增一个参数
                filter_img = torch.zeros_like(filter_img)

            # --- 实验 A: 高斯噪声 ---
            if self.noise_sigma > 0:
                noise = torch.randn_like(filter_img) * self.noise_sigma
                filter_img = filter_img + noise
                
            # --- 实验 B: 随机遮掩 (30% 面积) ---
            if self.apply_mask:
                filter_img = apply_random_mask(filter_img, mask_ratio=0.3)


                
            return raw_img, filter_img, label
# ================= 2. 模型结构优化 (B3支路 + 门控融合 + SE注意力) =================

class SEBlock(nn.Module):
    """
    通道注意力模块 (Squeeze-and-Excitation)
    用于自动学习 3072 个通道中哪些特征更重要
    """
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VGGBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): 
        return self.conv(x)

class DualBranchNet(nn.Module):
    def __init__(self, num_classes):
        super(DualBranchNet, self).__init__()
        # 支路1: 彩色原图提取
        self.backbone_raw = EfficientNet.from_pretrained('efficientnet-b3')
        # 支路2: 血管细节提取
        self.backbone_filter = EfficientNet.from_pretrained('efficientnet-b3')
        
        # 1. 门控组件：学习如何动态分配权重
        # 输入是两个支路的拼接 (1536*2=3072)，输出 1536 维的门控权重
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(3072, 1536, kernel_size=1),
            nn.Sigmoid()
        )
        # 2. SE 通道注意力模块
        self.se_module = SEBlock(3072)
        # 3. 融合后的卷积层
        self.fusion_vgg = nn.Sequential(
            VGGBlock(3072, 512),
            nn.MaxPool2d(2),
            VGGBlock(512, 256)
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, raw_x, filter_x):
        # 特征提取
        f1 = self.backbone_raw.extract_features(raw_x)    # [B, 1536, H, W]
        f2 = self.backbone_filter.extract_features(filter_x) # [B, 1536, H, W]
        
        # --- 步骤 1: 门控融合 (Gated Fusion) ---
        # 拼接后通过 Gate 计算注意力分配
        concat_features = torch.cat((f1, f2), dim=1)      # [B, 3072, H, W]
        gate_weight = self.gate(concat_features)          # [B, 1536, 1, 1]
        
        # 让模型决定原图看多少 (gate_weight)，血管图看多少 (1 - gate_weight)
        f1_gated = f1 * gate_weight
        f2_gated = f2 * (1.0 - gate_weight)
        
        # --- 步骤 2: 再次拼接并使用 SE 注意力 ---
        combined = torch.cat((f1_gated, f2_gated), dim=1) # [B, 3072, H, W]
        combined = self.se_module(combined)               # 自动增强有效通道
        
        # --- 步骤 3: 后续 VGG 层与分类 ---
        x = self.fusion_vgg(combined)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def mixup_data(raw, filt, y, alpha=0.2, device='cuda'):
    '''
    对双支路数据及其标签进行线性混合
    alpha: Beta分布的参数，常用 0.2 或 0.4
    '''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = raw.size()[0]
    # 生成随机打乱的索引
    index = torch.randperm(batch_size).to(device)

    # 对两个支路进行相同的打乱操作，确保原图和增强图依然是一对
    mixed_raw = lam * raw + (1 - lam) * raw[index, :]
    mixed_filt = lam * filt + (1 - lam) * filt[index, :]
    
    # 记录混合前的标签和打乱后的标签
    y_a, y_b = y, y[index]
    return mixed_raw, mixed_filt, y_a, y_b, lam

# ================= 3. 辅助函数 =================
def evaluate(loader, model, criterion, device):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    val_loss = 0.0
    with torch.no_grad():
        for raw, filt, labels in loader:
            raw, filt, labels = raw.to(device), filt.to(device), labels.to(device)
            outputs = model(raw, filt)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            probs = torch.softmax(outputs, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # --- 计算 AUC (针对多分类) ---
    # 将标签二值化 (例如 5 分类转为 [0,0,1,0,0] 这种 one-hot 形式)
    n_classes = len(np.unique(all_labels)) if len(np.unique(all_labels)) > 1 else 5 # 兜底防止只有一个类别
    lb = label_binarize(all_labels, classes=range(n_classes))
    
    # 计算 One-vs-Rest 的平均 AUC
    try:
        from sklearn.metrics import roc_auc_score
        # multi_class='ovr' 表示一对多，average='macro' 是各类别平均
        val_auc = roc_auc_score(lb, all_probs, multi_class='ovr', average='macro')
    except:
        val_auc = 0.0 # 防止某些极端样本情况下报错

    metrics = {
        "loss": val_loss / len(loader),
        "acc": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, average='macro', zero_division=0),
        "recall": recall_score(all_labels, all_preds, average='macro', zero_division=0),
        "f1": f1_score(all_labels, all_preds, average='macro'),
        "auc": val_auc
    }
    return metrics, all_labels, all_probs

def save_plots(history, labels, probs, classes, plot_dir):
    epochs = range(1, len(history['train_acc']) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1,2,1); plt.plot(epochs, history['train_loss'], label='Train'); plt.plot(epochs, history['val_loss'], label='Val'); plt.title('Loss'); plt.legend()
    plt.subplot(1,2,2); plt.plot(epochs, history['train_acc'], label='Train'); plt.plot(epochs, history['val_acc'], label='Val'); plt.title('Accuracy'); plt.legend()
    plt.savefig(os.path.join(plot_dir, 'training_curves.png'))
    
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(labels, np.argmax(probs, axis=1))
    cm_perc = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
    sns.heatmap(cm_perc, annot=True, fmt=".2%", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Normalized Confusion Matrix')
    plt.savefig(os.path.join(plot_dir, 'confusion_matrix.png'))
    
    plt.figure(figsize=(10, 8))
    lb = label_binarize(labels, classes=range(len(classes)))
    for i in range(len(classes)):
        fpr, tpr, _ = roc_curve(lb[:, i], probs[:, i])
        plt.plot(fpr, tpr, label=f'Class {i} (AUC = {auc(fpr, tpr):.4f})')
    plt.plot([0, 1], [0, 1], 'k--'); plt.legend(); plt.title('Multi-class ROC')
    plt.savefig(os.path.join(plot_dir, 'roc_curve.png'))
    plt.close('all')

# ================= 4. 主流程 =================
if __name__ == '__main__':
    RAW_ROOT = r'F:\DR\Data\aptos\aptos_split'
    FILTER_ROOT = r'F:\DR\Data\aptos\aptos_split_filter'
    BASE_SAVE_DIR = r"F:\DR\transfer_Learning\runs\train"
    
    # 如果想用 ImageNet 均值：
    RAW_MEAN, RAW_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    FILT_MEAN, FILT_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    
    # 或者如果想尝试本地均值（之前算出的那组）：
    # RAW_MEAN, RAW_STD = [0.3988, 0.2157, 0.0767], [0.2707, 0.1473, 0.0830]
    # FILT_MEAN, FILT_STD = [0.2172, 0.2172, 0.2172], [0.4124, 0.4124, 0.4124]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    save_path = os.path.join(BASE_SAVE_DIR, f"Dual_B3_VGG_Exp_{timestamp}")
    plot_path = os.path.join(save_path, "plot")
    os.makedirs(plot_path, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = {"img_size": 300, "batch_size": 8, "lr": 3e-5, "epochs": 100, "patience": 30}

    # --- 训练预处理 (关键修改：添加 RandomErasing) ---
    train_raw_trans = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),      
        transforms.Normalize(RAW_MEAN, RAW_STD),
        # 【添加位置】：在 Normalize 之后
        #transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0)
    ])

    train_filt_trans = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(FILT_MEAN, FILT_STD)
        # 血管支路建议保持完整，不加 Erasing
    ])

    # 验证预处理 (不加 Erasing)
    val_raw_trans = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(RAW_MEAN, RAW_STD)
    ])
    val_filt_trans = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(FILT_MEAN, FILT_STD)
    ])

    # 实例化 DataLoader
    train_ds = DualInputDataset(os.path.join(RAW_ROOT,'train'), os.path.join(FILTER_ROOT,'train'), 
                                raw_transform=train_raw_trans, filt_transform=train_filt_trans)
    val_ds = DualInputDataset(os.path.join(RAW_ROOT,'val'), os.path.join(FILTER_ROOT,'val'), 
                              raw_transform=val_raw_trans, filt_transform=val_filt_trans)
    test_ds = DualInputDataset(os.path.join(RAW_ROOT,'test'), os.path.join(FILTER_ROOT,'test'), 
                               raw_transform=val_raw_trans, filt_transform=val_filt_trans)
    
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False, num_workers=0)

    model = DualBranchNet(num_classes=5).to(device)
    
    # 分组参数：差异化学习率
    backbone_params = list(model.backbone_raw.parameters()) + list(model.backbone_filter.parameters())
    new_module_params = list(model.gate.parameters()) + list(model.se_module.parameters()) + \
                        list(model.fusion_vgg.parameters()) + list(model.fc.parameters())
    
    optimizer = optim.Adam([
        {'params': backbone_params, 'lr': config['lr']},           # 3e-5，预训练权重用较小学习率
        {'params': new_module_params, 'lr': config['lr'] * 3}      # 9e-5，新模块用3倍学习率
    ])
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    history = {'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[], 'val_precision':[], 'val_recall':[], 'val_f1':[]}
    best_acc, patience_counter = 0.0, 0

    # for epoch in range(1, config['epochs'] + 1):
    #     model.train()
    #     running_loss, running_acc = 0.0, 0.0
    #     train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{config['epochs']}")
        
    #     for raw, filt, labels in train_bar:
    #         raw, filt, labels = raw.to(device), filt.to(device), labels.to(device)
    #         # --- 【关键插入点 1: 数据混合】 ---
    #         # 只在训练阶段使用 Mixup，验证阶段严禁使用
    #         # alpha 建议设置在 0.2 到 0.4 之间
    #         #raw, filt, targets_a, targets_b, lam = mixup_data(raw, filt, labels, alpha=0.2, device=device)
            
    #         optimizer.zero_grad()
    #         outputs = model(raw, filt)
            
    #         # --- 【关键插入点 2: 损失函数计算】 ---
    #         # 此时的 Loss 需要按照混合比例 lam 分别计算两个标签的损失并相加
    #         #loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()
            
    #         running_loss += loss.item()
    #         running_acc += (outputs.argmax(1) == labels).sum().item() / labels.size(0)
    #         train_bar.set_postfix(loss=f"{loss.item():.4f}")

    #     val_m, _, _ = evaluate(val_loader, model, criterion, device)
    #     scheduler.step()
        
    #     history['train_loss'].append(running_loss / len(train_loader))
    #     history['train_acc'].append(running_acc / len(train_loader))
    #     history['val_loss'].append(val_m['loss'])
    #     history['val_acc'].append(val_m['acc'])
    #     history['val_precision'].append(val_m['precision'])
    #     history['val_recall'].append(val_m['recall'])
    #     history['val_f1'].append(val_m['f1'])
        
    #     print(f"\n[Epoch {epoch}] Val Acc: {val_m['acc']:.4f} | Val F1: {val_m['f1']:.4f}")
    #     pd.DataFrame(history).to_csv(os.path.join(save_path, "train_log.csv"), index=False)

    #     if val_m['acc'] > best_acc:
    #         best_acc = val_m['acc']
    #         torch.save(model.state_dict(), os.path.join(save_path, "best_model.pth"))
    #         patience_counter = 0
    #     else:
    #         patience_counter += 1
        
    #     if patience_counter >= config['patience']:
    #         break

    # 明确指向你已经训练好的模型权重
    PRETRAINED_MODEL_PATH = r"F:\DR\transfer_Learning\runs\train\Dual_B3_VGG_Exp_20260105_1132\best_model.pth"

    # 修改这里：直接加载这个路径
    model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH))
    model.eval()
    test_m, t_labels, t_probs = evaluate(test_loader, model, criterion, device)
    # save_plots(history, t_labels, t_probs, train_loader.dataset.classes, plot_path)
    print(f"完成！测试集准确率: {test_m['acc']:.4f}")
    # 记录最终配置与结果
    with open(os.path.join(save_path, "config.txt"), "w") as f:
        f.write(f"Model: Dual-branch EfficientNet-B3 + VGGBlock\n")
        f.write(f"Img Size: {config['img_size']}\n")
        f.write(f"Final Test Acc: {test_m['acc']:.4f}\n")
        f.write(f"Final Test F1: {test_m['f1']:.4f}\n")

    print(f"所有任务已完成！结果路径: {save_path}")

# --- 鲁棒性实验：高斯噪声测试 ---
    '''
    print("\n" + "="*30)
    print("开始执行鲁棒性实验：血管图噪声干扰测试")
    print("="*30)
    
    sigma_levels = [0.0, 0.05, 0.1, 0.2, 0.5] 
    robustness_results = []

    # 关键修改：再次确保从原始预训练路径加载，而不是从空的 save_path 加载
    model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH)) 
    model.eval()

    for sigma in sigma_levels:
        # 创建带特定噪声水平的测试集
        noise_test_ds = DualInputDataset(
            os.path.join(RAW_ROOT,'test'), 
            os.path.join(FILTER_ROOT,'test'), 
            raw_transform=val_raw_trans, 
            filt_transform=val_filt_trans,
            noise_sigma=sigma
        )
        noise_loader = DataLoader(noise_test_ds, batch_size=config['batch_size'], shuffle=False)
        
        # 执行评估
        metrics, _, _ = evaluate(noise_loader, model, criterion, device)
        
        print(f"Sigma: {sigma:<5} | Acc: {metrics['acc']:.4f} | F1: {metrics['f1']:.4f} | AUC: {metrics['auc']:.4f}")
        
        # 记录到结果列表中（这些键名将成为 CSV 的表头）
        robustness_results.append({
            "sigma": sigma, 
            "val_acc": metrics["acc"], 
            "val_precision": metrics["precision"],
            "val_recall": metrics["recall"],
            "val_f1": metrics["f1"],
            "val_auc": metrics["auc"]
        })

    # 将实验结果保存为 CSV 方便画图
    robust_df = pd.DataFrame(robustness_results)
    robust_df.to_csv(os.path.join(save_path, "robustness_noise_test.csv"), index=False)
    print(f"鲁棒性实验完成！结果已保存至: {save_path}")
    


    # --- 鲁棒性实验 B：30% 随机遮掩测试 ---
    print("\n" + "="*30)
    print("开始执行鲁棒性实验 B：血管图 30% 随机遮掩测试")
    print("="*30)

    # 创建遮掩测试集
    mask_test_ds = DualInputDataset(
        os.path.join(RAW_ROOT, 'test'), 
        os.path.join(FILTER_ROOT, 'test'), 
        raw_transform=val_raw_trans, 
        filt_transform=val_filt_trans,
        apply_mask=True  # 开启遮掩
    )
    mask_loader = DataLoader(mask_test_ds, batch_size=config['batch_size'], shuffle=False)

    # 重新加载模型最佳权重
    model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH))
    model.eval()

    # 执行评估
    m_mask, _, _ = evaluate(mask_loader, model, criterion, device)

    # 打印全指标结果
    print(f"【30% Occlusion 结果】")
    print(f"Acc: {m_mask['acc']:.4f} | Precision: {m_mask['precision']:.4f} | "
          f"Recall: {m_mask['recall']:.4f} | F1: {m_mask['f1']:.4f} | AUC: {m_mask['auc']:.4f}")

    # 保存遮掩实验结果到单独的 CSV
    mask_results = [{
        "experiment": "30%_Random_Masking",
        "val_acc": m_mask["acc"],
        "val_precision": m_mask["precision"],
        "val_recall": m_mask["recall"],
        "val_f1": m_mask["f1"],
        "val_auc": m_mask["auc"]
    }]
    pd.DataFrame(mask_results).to_csv(os.path.join(save_path, "robustness_mask_test.csv"), index=False)
    print(f"遮掩实验完成！结果已保存。")


    # --- 鲁棒性实验 C：形态学腐蚀测试 (5x5 Kernel) ---
    print("\n" + "="*30)
    print("开始执行鲁棒性实验 C：血管图形态学腐蚀测试")
    print("="*30)

    # 创建腐蚀测试集
    erode_test_ds = DualInputDataset(
        os.path.join(RAW_ROOT, 'test'), 
        os.path.join(FILTER_ROOT, 'test'), 
        raw_transform=val_raw_trans, 
        filt_transform=val_filt_trans,
        apply_erode=True  # 开启腐蚀
    )
    erode_loader = DataLoader(erode_test_ds, batch_size=config['batch_size'], shuffle=False)

    # 确保加载最佳模型
    model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH))
    model.eval()

    # 执行评估
    m_erode, _, _ = evaluate(erode_loader, model, criterion, device)

    # 打印结果
    print(f"【Erosion 5x5 结果】")
    print(f"Acc: {m_erode['acc']:.4f} | Precision: {m_erode['precision']:.4f} | "
          f"Recall: {m_erode['recall']:.4f} | F1: {m_erode['f1']:.4f} | AUC: {m_erode['auc']:.4f}")

    # 保存实验结果
    erode_results = [{
        "experiment": "Morphological_Erosion_5x5",
        "val_acc": m_erode["acc"],
        "val_precision": m_erode["precision"],
        "val_recall": m_erode["recall"],
        "val_f1": m_erode["f1"],
        "val_auc": m_erode["auc"]
    }]
    pd.DataFrame(erode_results).to_csv(os.path.join(save_path, "robustness_erode_test.csv"), index=False)
    print(f"腐蚀实验完成！结果已保存至: {save_path}")
    '''

# --- 鲁棒性实验 D：血管图完全缺失压力测试 (100% Zero-out) ---
    print("\n" + "="*30)
    print("开始执行鲁棒性实验 D：血管图 100% 缺失压力测试")
    print("="*30)

    # 创建完全缺失测试集
    zero_test_ds = DualInputDataset(
        os.path.join(RAW_ROOT, 'test'), 
        os.path.join(FILTER_ROOT, 'test'), 
        raw_transform=val_raw_trans, 
        filt_transform=val_filt_trans,
        apply_zero=True  # 开启全黑模式
    )
    zero_loader = DataLoader(zero_test_ds, batch_size=config['batch_size'], shuffle=False)

    # 加载权重并评估
    model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH))
    model.eval()

    m_zero, _, _ = evaluate(zero_loader, model, criterion, device)

    # 打印结果
    print(f"【100% Zero-out 压力测试结果】")
    print(f"Acc: {m_zero['acc']:.4f} | Precision: {m_zero['precision']:.4f} | "
          f"Recall: {m_zero['recall']:.4f} | F1: {m_zero['f1']:.4f} | AUC: {m_zero['auc']:.4f}")

    # 保存实验结果
    zero_results = [{
        "experiment": "Total_Input_Loss_100%",
        "val_acc": m_zero["acc"],
        "val_precision": m_zero["precision"],
        "val_recall": m_zero["recall"],
        "val_f1": m_zero["f1"],
        "val_auc": m_zero["auc"]
    }]
    pd.DataFrame(zero_results).to_csv(os.path.join(save_path, "robustness_zero_test.csv"), index=False)
    print(f"极端鲁棒性实验完成！结果已保存至: {save_path}")
