# HW1: 从零开始构建三层神经网络分类器

基于 NumPy 从零实现的三层 MLP 分类器，用于 EuroSAT 遥感图像数据集的地表覆盖分类。

## 任务概述

手工搭建三层神经网络（MLP）分类器，在遥感图像数据集 EuroSAT 上进行训练，实现基于卫星图像的土地覆盖分类。EuroSAT 包含 10 个类别：AnnualCrop、Forest、HerbaceousVegetation、Highway、Industrial、Pasture、PermanentCrop、Residential、River、SeaLake。

**关键约束**：不允许使用 PyTorch、TensorFlow 等自动微分框架，需自主实现前向传播、反向传播和优化器。

## 环境依赖

- Python 3.8+
- numpy >= 1.20
- pillow >= 8.0
- matplotlib >= 3.3
- seaborn >= 0.11
- scikit-learn >= 0.24

安装命令：
```bash
pip install numpy pillow matplotlib seaborn scikit-learn
```

## 数据集

1. 下载 EuroSAT 数据集（RGB 版本）
2. 将数据放置在 `data/EuroSAT_RGB` 目录下，目录结构如下：
```
data/EuroSAT_RGB/
├── AnnualCrop/
├── Forest/
├── HerbaceousVegetation/
├── Highway/
├── Industrial/
├── Pasture/
├── PermanentCrop/
├── Residential/
├── River/
└── SeaLake/
```

## 代码结构

```
src/
├── data_loader.py        # 数据加载与预处理
├── layers.py             # 线性层、激活函数、Softmax交叉熵损失
├── model.py              # 三层MLP模型定义
├── optimizer.py          # SGD优化器（含动量和学习率衰减）
├── train.py              # 训练脚本
├── test.py               # 测试脚本
├── hyperparam_search.py  # 超参数网格搜索
└── utils.py              # 可视化工具函数
```

## 快速开始

### 训练模型

**使用默认配置（ReLU, hidden_dim=256）**：
```bash
python src/train.py --data_dir data/EuroSAT_RGB --hidden_dim 256 --activation relu --lr 0.01 --l2_lambda 0.001 --epochs 30 --batch_size 256
```

**使用最佳配置（Tanh, hidden_dim=512）**：
```bash
python src/train.py --data_dir data/EuroSAT_RGB --hidden_dim 512 --activation tanh --lr 0.01 --l2_lambda 0.0 --epochs 30 --batch_size 256
```

### 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_dir` | `./data/EuroSAT_RGB` | 数据集路径 |
| `--hidden_dim` | 256 | 隐藏层维度 |
| `--activation` | `relu` | 激活函数 (relu/sigmoid/tanh) |
| `--lr` | 0.01 | 初始学习率 |
| `--l2_lambda` | 0.0 | L2正则化系数 |
| `--momentum` | 0.9 | 动量系数 |
| `--lr_decay` | 0.001 | 学习率衰减率 |
| `--batch_size` | 64 | 批大小 |
| `--epochs` | 50 | 训练轮数 |
| `--save_path` | `./results/best_model` | 模型权重保存路径 |
| `--output_dir` | `./results` | 输出目录 |

### 测试模型

```bash
python src/test.py --data_dir data/EuroSAT_RGB --model_path results/best_model_tanh512.npz --hidden_dim 512 --activation tanh
```

### 超参数搜索

```bash
cd src
python hyperparam_search.py --data_dir ../data/EuroSAT_RGB
```

## 实验结果

- **最佳测试准确率**：63.78%（Tanh, hidden_dim=512）
- **最佳类别**：Forest (F1=0.84), Industrial (F1=0.81)

## GitHub 仓库

https://github.com/ZhenYongWei/fd-hw