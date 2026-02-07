import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 1. 构造数据
data = {
    'val_acc': [0.904185022, 0.90969163, 0.921806167, 0.919603524, 0.911894273, 0.912995595, 0.9030837, 0.922907489, 0.91629956],
    'val_f1': [0.892578602, 0.899318537, 0.912857638, 0.909622948, 0.901175243, 0.903248615, 0.89030245, 0.913558895, 0.906263335],
    'val_auc': [0.980097553, 0.980556444, 0.985260953, 0.984376552, 0.984700618, 0.983888329, 0.981721745, 0.986059596, 0.984409923],
    'param_name': ['img_size', 'img_size', 'img_size', 'lr', 'lr', 'lr', 'batch_size', 'batch_size', 'batch_size'],
    'param_value': [224, 256, 300, 0.00003, 0.00009, 0.00015, 2, 4, 8]
}

df = pd.DataFrame(data)

# 2. 设置绘图风格
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=100)
metrics = ['val_acc', 'val_f1', 'val_auc']
params = ['img_size', 'lr', 'batch_size']
titles = ['Image Size', 'Learning Rate', 'Batch Size']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # 对应三个指标的颜色

# 3. 循环生成子图
for i, param in enumerate(params):
    subset = df[df['param_name'] == param].copy()
    
    # 将参数值转为字符串，确保横轴等间距显示，避免数值跨度导致挤压
    subset['param_value_str'] = subset['param_value'].astype(str)
    
    for j, metric in enumerate(metrics):
        sns.lineplot(
            ax=axes[i],
            data=subset,
            x='param_value_str',
            y=metric,
            marker='o',
            markersize=8,
            label=metric,
            color=colors[j],
            linewidth=2
        )
    
    axes[i].set_title(f'Performance vs {titles[i]}', fontsize=14, fontweight='bold')
    axes[i].set_xlabel(titles[i], fontsize=12)
    axes[i].set_ylabel('Metric Score', fontsize=12)
    axes[i].legend(title='Metrics', frameon=True)
    
    # 细节优化：限制纵轴范围，让波动更明显
    axes[i].set_ylim(subset[metrics].min().min() - 0.01, 1.0) 

plt.tight_layout()

# 4. 保存并显示
# plt.savefig('parameter_analysis.png', bbox_inches='tight')
plt.show()