# 驾驶员分心驾驶行为检测系统

基于 PyTorch 迁移学习的驾驶员行为图像分类项目，使用 State Farm Distracted Driver Detection 数据集（10 类驾驶行为），训练 MobileNetV3-Small 模型，并用 Gradio 搭建了一个可交互的检测网页：上传驾驶员图片后，输出最可能的 3 种行为及概率，并判断属于安全驾驶还是危险驾驶。

## 行为类别（10 类）

| 类别 | 含义 |
| --- | --- |
| c0 | 安全驾驶 |
| c1 | 右手发短信 |
| c2 | 右手打电话 |
| c3 | 左手发短信 |
| c4 | 左手打电话 |
| c5 | 操作收音机 |
| c6 | 喝水 |
| c7 | 伸手拿东西 |
| c8 | 整理头发或化妆 |
| c9 | 与乘客交谈 |

## 技术要点

- **模型**：MobileNetV3-Small（ImageNet 预训练）迁移学习 — 冻结大部分特征层，只微调最后一个特征块与重新设计的分类头（256 维全连接 + Hardswish + Dropout 0.3）
- **训练**：图片统一 224×224；训练集做随机裁剪、水平翻转、颜色抖动、随机擦除等增强；按 8:2 划分训练/验证集
- **优化**：AdamW（lr 3e-4，weight decay 1e-2）+ 余弦退火学习率；交叉熵损失（label smoothing 0.1）；混合精度训练（AMP）；梯度裁剪；验证准确率不提升时早停（patience 4），最长 15 个 epoch
- **推理应用**：Gradio 网页界面，加载训练好的 `best_model.pth`，展示 Top-3 概率条与安全/危险状态卡片

## 文件说明

| 文件 | 说明 |
| --- | --- |
| `train.py` | 训练脚本：数据处理、模型微调、训练与验证、保存最佳权重 |
| `app.py` | Gradio 检测应用：加载权重并提供网页交互界面 |
| `best_model.pth` | 训练得到的最优权重（含 `model_state_dict`、类别名与最佳验证准确率） |
| `requirements.txt` | 运行依赖（torch、torchvision、gradio、Pillow） |

## 运行方式

```bash
pip install -r requirements.txt

# 训练（需先把数据集放到 data/imgs/train/，按类别分子文件夹）
python train.py

# 启动检测网页（同目录下需有 best_model.pth）
python app.py
```

> 数据集来自 Kaggle 的 State Farm Distracted Driver Detection 竞赛，出于体积与版权原因未包含在本仓库中。
