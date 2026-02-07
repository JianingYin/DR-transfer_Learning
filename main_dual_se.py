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
# ================= 1. 数据对齐 Dataset =================
class DualInputDataset(Dataset):
    """
    通过相对路径实现原图(Raw)与增强图(Filter)的一一对应
    """
    def __init__(self, raw_dir, filter_dir, raw_transform=None, filt_transform=None):
        self.raw_dataset = datasets.ImageFolder(raw_dir)
        self.filter_dir = filter_dir
        self.raw_transform = raw_transform  # 确保这里是 raw_transform
        self.filt_transform = filt_transform # 确保这里是 filt_transform
        self.classes = self.raw_dataset.classes

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, index):
        raw_img_path, label = self.raw_dataset.samples[index]
        # 获取相对路径以在 Filter 文件夹中寻找同名文件
        rel_path = os.path.relpath(raw_img_path, self.raw_dataset.root)
        filter_img_path = os.path.join(self.filter_dir, rel_path)
        
        raw_img = Image.open(raw_img_path).convert('RGB')
        # 增加容错：如果 filter 文件丢失，报错提醒
        if not os.path.exists(filter_img_path):
            raise FileNotFoundError(f"Filter image not found: {filter_img_path}")
        filter_img = Image.open(filter_img_path).convert('RGB')

        # 分别应用不同的变换
        if self.raw_transform:
            raw_img = self.raw_transform(raw_img)
        if self.filt_transform:
            filter_img = self.filt_transform(filter_img)

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
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
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
            
    metrics = {
        "loss": val_loss / len(loader),
        "acc": accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds, average='macro'),
        "precision": precision_score(all_labels, all_preds, average='macro', zero_division=0),
        "recall": recall_score(all_labels, all_preds, average='macro', zero_division=0)
    }
    return metrics, all_labels, np.array(all_probs)

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
    RAW_ROOT = r'/root/DR/transfer_Learning/Data/aptos/aptos_split/'
    FILTER_ROOT = r'/root/DR/transfer_Learning/Data/aptos/aptos_split_filter/'
    BASE_SAVE_DIR = r"/root/DR/transfer_Learning/runs/train/"
    
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
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0)
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

    for epoch in range(1, config['epochs'] + 1):
        model.train()
        running_loss, running_acc = 0.0, 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{config['epochs']}")
        
        for raw, filt, labels in train_bar:
            raw, filt, labels = raw.to(device), filt.to(device), labels.to(device)
            # --- 【关键插入点 1: 数据混合】 ---
            # 只在训练阶段使用 Mixup，验证阶段严禁使用
            # alpha 建议设置在 0.2 到 0.4 之间
            raw, filt, targets_a, targets_b, lam = mixup_data(raw, filt, labels, alpha=0.2, device=device)
            
            optimizer.zero_grad()
            outputs = model(raw, filt)
            
            # --- 【关键插入点 2: 损失函数计算】 ---
            # 此时的 Loss 需要按照混合比例 lam 分别计算两个标签的损失并相加
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            running_acc += (outputs.argmax(1) == targets_a).sum().item() / labels.size(0)
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        val_m, _, _ = evaluate(val_loader, model, criterion, device)
        scheduler.step()
        
        history['train_loss'].append(running_loss / len(train_loader))
        history['train_acc'].append(running_acc / len(train_loader))
        history['val_loss'].append(val_m['loss'])
        history['val_acc'].append(val_m['acc'])
        history['val_precision'].append(val_m['precision'])
        history['val_recall'].append(val_m['recall'])
        history['val_f1'].append(val_m['f1'])
        
        print(f"\n[Epoch {epoch}] Val Acc: {val_m['acc']:.4f} | Val F1: {val_m['f1']:.4f}")
        pd.DataFrame(history).to_csv(os.path.join(save_path, "train_log.csv"), index=False)

        if val_m['acc'] > best_acc:
            best_acc = val_m['acc']
            torch.save(model.state_dict(), os.path.join(save_path, "best_model.pth"))
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= config['patience']:
            break

    model.load_state_dict(torch.load(os.path.join(save_path, "best_model.pth")))
    test_m, t_labels, t_probs = evaluate(test_loader, model, criterion, device)
    save_plots(history, t_labels, t_probs, train_loader.dataset.classes, plot_path)
    print(f"完成！测试集准确率: {test_m['acc']:.4f}")
    # 记录最终配置与结果
    with open(os.path.join(save_path, "config.txt"), "w") as f:
        f.write(f"Model: Dual-branch EfficientNet-B3 + VGGBlock\n")
        f.write(f"Img Size: {config['img_size']}\n")
        f.write(f"Final Test Acc: {test_m['acc']:.4f}\n")
        f.write(f"Final Test F1: {test_m['f1']:.4f}\n")

    print(f"所有任务已完成！结果路径: {save_path}")