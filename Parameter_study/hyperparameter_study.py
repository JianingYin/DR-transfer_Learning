# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from PIL import Image
from efficientnet_pytorch import EfficientNet
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, roc_curve, auc)
from sklearn.preprocessing import label_binarize

# 导入你之前定义的类 (假设在同一个文件或已定义)
class DualInputDataset(Dataset):
    def __init__(self, raw_dir, filter_dir, transform=None):
        self.raw_dataset = datasets.ImageFolder(raw_dir)
        self.filter_dir = filter_dir
        self.transform = transform

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, index):
        raw_img_path, label = self.raw_dataset.samples[index]
        # 根据原图路径推导血管图路径
        rel_path = os.path.relpath(raw_img_path, self.raw_dataset.root)
        filter_img_path = os.path.join(self.filter_dir, rel_path)
        
        raw_img = Image.open(raw_img_path).convert('RGB')
        filter_img = Image.open(filter_img_path).convert('RGB')

        if self.transform:
            raw_img = self.transform(raw_img)
            filter_img = self.transform(filter_img)
            
        return raw_img, filter_img, label

# ================= 1. 核心模型 =================
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
            
    # AUC 计算
    all_probs = np.array(all_probs)
    auc_score = 0.0
    try:
        unique_classes = np.unique(all_labels)
        if len(unique_classes) > 1:
            lb = label_binarize(all_labels, classes=range(5)) # 假设 5 分类
            auc_score = roc_auc_score(lb, all_probs, average='macro', multi_class='ovr')
    except Exception as e:
        print(f"AUC计算错误: {e}")
    
    metrics = {
        "loss": val_loss / len(loader),
        "val_acc": accuracy_score(all_labels, all_preds), # 统一键名
        "val_f1": f1_score(all_labels, all_preds, average='macro'),
        "val_precision": precision_score(all_labels, all_preds, average='macro', zero_division=0),
        "val_recall": recall_score(all_labels, all_preds, average='macro', zero_division=0),
        "val_auc": auc_score
    }
    return metrics

# ================= 2. 实验配置与队列 =================

if __name__ == '__main__':
    # 基础路径配置
    DATA_ROOTS = {"raw": r"F:\DR\Data\aptos\aptos_split", "filt": r"F:\DR\Data\aptos\aptos_split_filter"}
    BASE_SAVE = r"F:\DR\transfer_Learning\Parameter_study"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # --- 默认基准参数 (Baseline) ---
    baseline_config = {
        "img_size": 256,
        "lr": 3e-5,
        "batch_size": 4,
        "epochs": 30 # 对比实验通常不需要跑太久，30轮足以看出趋势
    }

    # --- 定义实验队列 ---
    # 格式: (实验组名, 改变的参数名, 测试数值列表)
    experiments = [
        ("Size_Study", "img_size", [224, 256, 300]),
        ("LR_Study", "lr", [3e-5, 9e-5, 1.5e-4]),
        ("Batch_Study", "batch_size", [2, 4, 8])
    ]

    all_hyper_results = []

    for group_name, param_key, values in experiments:
        print(f"\n{'='*20} 开始实验组: {group_name} {'='*20}")
        
        for val in values:
            # 复制基准配置并修改当前变量
            current_cfg = baseline_config.copy()
            current_cfg[param_key] = val
            
            exp_name = f"{group_name}_{param_key}_{val}"
            print(f"\n>>> 正在运行: {exp_name}")
            
            save_path = os.path.join(BASE_SAVE, group_name, exp_name)
            os.makedirs(save_path, exist_ok=True)

            # 1. 动态调整 Transform (针对图像尺寸实验)
            transform = transforms.Compose([
                transforms.Resize((current_cfg["img_size"], current_cfg["img_size"])),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            # 2. 准备数据加载器
            train_ds = DualInputDataset(os.path.join(DATA_ROOTS["raw"],'train'), os.path.join(DATA_ROOTS["filt"],'train'), transform)
            val_ds = DualInputDataset(os.path.join(DATA_ROOTS["raw"],'val'), os.path.join(DATA_ROOTS["filt"],'val'), transform)
            
            train_loader = DataLoader(train_ds, batch_size=current_cfg["batch_size"], shuffle=True, num_workers=0)
            val_loader = DataLoader(val_ds, batch_size=current_cfg["batch_size"], shuffle=False, num_workers=0)

            # 3. 初始化模型 (Full 结构)
            model = DualBranchNet(num_classes=5).to(device)
            optimizer = optim.Adam(model.parameters(), lr=current_cfg["lr"])
            criterion = nn.CrossEntropyLoss()

            best_acc = 0.0
            best_metrics = None

            # 4. 训练循环
            for epoch in range(1, current_cfg["epochs"] + 1):
                model.train()
                for raw, filt, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{current_cfg['epochs']}"):
                    raw, filt, labels = raw.to(device), filt.to(device), labels.to(device)
                    optimizer.zero_grad()
                    loss = criterion(model(raw, filt), labels)
                    loss.backward()
                    optimizer.step()

                # 验证
                m = evaluate(val_loader, model,criterion, device)
                if m['val_acc'] > best_acc:
                    best_acc = m['val_acc']
                    best_metrics = m.copy()
                    torch.save(model.state_dict(), os.path.join(save_path, "best_hyper.pth"))
                
                print(f"Current Acc: {m['val_acc']:.4f} | Best: {best_acc:.4f}")

            # 5. 记录结果
            best_metrics.update({
                "exp_group": group_name,
                "param_name": param_key,
                "param_value": val,
                "img_size": current_cfg["img_size"],
                "lr": current_cfg["lr"],
                "batch_size": current_cfg["batch_size"]
            })
            all_hyper_results.append(best_metrics)
            # 每跑完一个子实验就覆盖保存一次，防止程序中途崩溃导致数据丢失
# ---【新增：保险代码】每跑完一个子实验就存一次 CSV ---
            temp_df = pd.DataFrame(all_hyper_results)
            temp_df.to_csv(os.path.join(BASE_SAVE, "hyperparameter_sync_backup.csv"), index=False)
            
            # 及时释放显存
            del model, optimizer, train_loader, val_loader
            torch.cuda.empty_cache()

    # 汇总保存
    final_df = pd.DataFrame(all_hyper_results)
    final_df.to_csv(os.path.join(BASE_SAVE, "hyperparameter_comparison.csv"), index=False)
    print("\n[FINISH] 所有超参数对比实验已完成！")