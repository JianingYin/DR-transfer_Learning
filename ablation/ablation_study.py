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
                             f1_score, roc_auc_score)
from sklearn.preprocessing import label_binarize
from efficientnet_pytorch import EfficientNet
from tqdm import tqdm
import matplotlib.pyplot as plt

# ================= 1. 数据集定义 =================

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

# ================= 2. 支持消融的动态模型 =================

class SEBlock(nn.Module):
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
        return x * y

class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VGGBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class AblationNet(nn.Module):
    def __init__(self, num_classes=5, exp_type="full"):
        super(AblationNet, self).__init__()
        self.exp_type = exp_type
        
        # 支路1: 原图
        self.backbone_raw = EfficientNet.from_pretrained('efficientnet-b3')
        
        # 支路2: 血管图 (Exp 1 消融)
        if exp_type != "no_vessel":
            self.backbone_filter = EfficientNet.from_pretrained('efficientnet-b3')
            # 门控模块 (Exp 2 消融)
            if exp_type != "no_gated":
                self.gate = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(3072, 1536, kernel_size=1),
                    nn.Sigmoid()
                )

        self.combined_chs = 1536 if exp_type == "no_vessel" else 3072

        # Exp 3: SE 通道注意力消融
        if exp_type != "no_se":
            self.se_module = SEBlock(self.combined_chs)

        # Exp 4: VGG 整合消融
        if exp_type != "no_vgg":
            self.fusion_vgg = nn.Sequential(
                VGGBlock(self.combined_chs, 512), 
                nn.MaxPool2d(2), 
                VGGBlock(512, 256)
            )
            self.final_chs = 256
        else:
            self.final_chs = self.combined_chs

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.final_chs, num_classes)

    def forward(self, raw_x, filter_x):
        f1 = self.backbone_raw.extract_features(raw_x)
        
        if self.exp_type == "no_vessel":
            combined = f1
        else:
            f2 = self.backbone_filter.extract_features(filter_x)
            if self.exp_type == "no_gated":
                combined = torch.cat((f1, f2), dim=1)
            else:
                gate_w = self.gate(torch.cat((f1, f2), dim=1))
                combined = torch.cat((f1 * gate_w, f2 * (1.0 - gate_w)), dim=1)

        if hasattr(self, 'se_module'): combined = self.se_module(combined)
        if hasattr(self, 'fusion_vgg'): x = self.fusion_vgg(combined)
        else: x = combined

        return self.fc(self.avgpool(x).view(x.size(0), -1))

# ================= 3. 评估逻辑 =================

def evaluate(loader, model, device):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for raw, filt, labels in loader:
            raw, filt, labels = raw.to(device), filt.to(device), labels.to(device)
            outputs = model(raw, filt)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    # 针对多分类计算 AUC
    lb = label_binarize(all_labels, classes=range(all_probs.shape[1]))
    
    return {
        "val_acc": accuracy_score(all_labels, all_preds),
        "val_precision": precision_score(all_labels, all_preds, average='macro', zero_division=0),
        "val_recall": recall_score(all_labels, all_preds, average='macro', zero_division=0),
        "val_f1": f1_score(all_labels, all_preds, average='macro'),
        "val_auc": roc_auc_score(lb, all_probs, multi_class='ovr', average='macro')
    }

# ================= 4. 主循环执行 =================

if __name__ == '__main__':
    DATA_ROOTS = {"raw": r"F:\DR\Data\aptos\aptos_split", "filt": r"I:\Nico\DR\Data\aptos\aptos_split_filter"}
    BASE_SAVE = r"F:\DR\transfer_Learning\ablation"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = {"img_size": 300, "batch_size": 8, "lr": 3e-5, "epochs": 50}

    transform = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    tasks = [
        ("Full_Model", "full"),
        ("Exp1_NoVessel", "no_vessel"),
        ("Exp2_NoGated", "no_gated"),
        ("Exp3_NoSE", "no_se"),
        ("Exp4_NoVGG", "no_vgg")
    ]

    summary_data = []

    for name, mode in tasks:
        print(f"\n{'#'*40}\n>>> 启动任务: {name}\n{'#'*40}")
        exp_path = os.path.join(BASE_SAVE, name)
        os.makedirs(exp_path, exist_ok=True)

        train_ds = DualInputDataset(os.path.join(DATA_ROOTS["raw"],'train'), os.path.join(DATA_ROOTS["filt"],'train'), transform)
        val_ds = DualInputDataset(os.path.join(DATA_ROOTS["raw"],'val'), os.path.join(DATA_ROOTS["filt"],'val'), transform)
        train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False, num_workers=0)

        model = AblationNet(num_classes=5, exp_type=mode).to(device)
        optimizer = optim.Adam(model.parameters(), lr=config['lr'])
        criterion = nn.CrossEntropyLoss()

        best_acc, history = 0.0, []
        best_metrics = None

        for epoch in range(1, config['epochs'] + 1):
            model.train()
            total_loss = 0
            for raw, filt, labels in tqdm(train_loader, desc=f"{name} Ep {epoch}"):
                raw, filt, labels = raw.to(device), filt.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = criterion(model(raw, filt), labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            m = evaluate(val_loader, model, device)
            print(f"[{name}] Ep{epoch} Acc: {m['val_acc']:.4f} F1: {m['val_f1']:.4f}")
            
            m.update({"epoch": epoch, "loss": total_loss/len(train_loader)})
            history.append(m)

            if m['val_acc'] > best_acc:
                best_acc = m['val_acc']
                torch.save(model.state_dict(), os.path.join(exp_path, "best_model.pth"))
                best_metrics = m.copy()

        pd.DataFrame(history).to_csv(os.path.join(exp_path, "train_log.csv"), index=False)
        best_metrics["Experiment"] = name
        summary_data.append(best_metrics)

    # 保存最终结果
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(BASE_SAVE, "ablation_final_summary.csv"), index=False)
    
    # 简单的柱状图可视化
    plt.figure(figsize=(10, 6))
    plt.bar(summary_df['Experiment'], summary_df['val_acc'], color='skyblue')
    plt.title('Ablation Study: Validation Accuracy')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=15)
    plt.ylim(0, 1.0)
    plt.savefig(os.path.join(BASE_SAVE, "ablation_visual.png"))
    
    print("\n[DONE] 所有消融实验已完成，结果已保存。")