#1.库导入区 导入训练所需的 Python 库，包括 PyTorch、TorchVision、路径处理、多进程支持等。
import multiprocessing
import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from torch.cuda.amp import autocast, GradScaler

from torch.utils.data import DataLoader, random_split

from torchvision import datasets, transforms, models


def main():
#2.参数配置区  设置数据集路径、训练设备、批大小、训练轮数、学习率和模型保存路径。
    DATA_DIR = Path("data/imgs/train")

    if not DATA_DIR.exists():
        DATA_DIR = Path(
            r"C:\Users\31292\Desktop\state-farm-distracted-driver-detection\imgs\train"
        )

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BATCH_SIZE = 64
    EPOCHS = 15
    LR = 3e-4

    SAVE_PATH = "best_model.pth"

    print("=" * 50)
    print("Device:", DEVICE)
    print("Dataset:", DATA_DIR)
    print("=" * 50)

#3.数据增强与预处理区 训练集进行随机增强，验证集只做固定处理，保证验证结果稳定。
    #训练集处理
    train_tf = transforms.Compose([

        transforms.RandomResizedCrop(
            224,
            scale=(0.8, 1.0)
        ),

        transforms.RandomHorizontalFlip(),

        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),

        transforms.ToTensor(),

        transforms.RandomErasing(p=0.2),

        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    #验证集处理
    val_tf = transforms.Compose([

        transforms.Resize((224, 224)),

        transforms.ToTensor(),

        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

#4.数据集读取与划分区  读取本地图片数据集，自动获取类别名称，并按照 8:2 划分训练集和验证集。
    full_dataset = datasets.ImageFolder(
        DATA_DIR,
        transform=train_tf
    )

    class_names = full_dataset.classes
    num_classes = len(class_names)

    print("Classes:", class_names)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_set, val_set = random_split(
        full_dataset,
        [train_size, val_size]
    )

    val_set.dataset = copy.deepcopy(full_dataset)
    val_set.dataset.transform = val_tf

#5. 数据加载器区  把数据集封装成批量加载器，提高训练效率。
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

#6. CNN 模型构建与迁移学习区 加载 MobileNetV3-Small 预训练模型，冻结大部分参数，只微调最后特征层和分类头。
    model = models.mobilenet_v3_small(
        weights=models.MobileNet_V3_Small_Weights.DEFAULT
    )

    for param in model.parameters():
        param.requires_grad = False

    for param in model.features[-1].parameters():
        param.requires_grad = True

    in_features = model.classifier[3].in_features

    model.classifier[3] = nn.Sequential(

        nn.Linear(in_features, 256),
        nn.Hardswish(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )

    for param in model.classifier.parameters():
        param.requires_grad = True

    model = model.to(DEVICE)

#7. 损失函数与优化器区 设置分类损失函数和参数更新方式。
    criterion = nn.CrossEntropyLoss(
        label_smoothing=0.1
    )

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=1e-2
    )

#8. 训练优化策略区 提高训练速度和稳定性，防止过拟合。
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )

    scaler = GradScaler()

    best_acc = 0.0

    patience = 4
    trigger_times = 0

#9. 模型训练区 完成前向传播、损失计算、反向传播和参数更新。
    for epoch in range(EPOCHS):

        model.train()

        train_correct = 0
        train_total = 0
        train_loss = 0.0

        for imgs, labels in train_loader:

            imgs = imgs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()

            with autocast():

                outputs = model(imgs)

                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            scaler.step(optimizer)

            scaler.update()

            train_loss += loss.item()

            preds = outputs.argmax(1)

            train_correct += (preds == labels).sum().item()

            train_total += labels.size(0)

        scheduler.step()

        train_acc = train_correct / train_total

#10. 验证评估与最佳模型保存区

        model.eval()

        val_correct = 0
        val_total = 0
    #对应代码
        with torch.no_grad():

            for imgs, labels in val_loader:

                imgs = imgs.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

                outputs = model(imgs)

                preds = outputs.argmax(1)

                val_correct += (preds == labels).sum().item()

                val_total += labels.size(0)

        val_acc = val_correct / val_total

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

    #保存最佳模型
        if val_acc > best_acc:

            best_acc = val_acc

            trigger_times = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "best_acc": best_acc
                },
                SAVE_PATH
            )

            print("✔ Best Model Saved")

        else:
            trigger_times += 1

        if trigger_times >= patience:

            print("\nEarly Stopping Triggered")

            break

    print("\n" + "=" * 50)
    print("Training Finished")
    print("Best Validation Accuracy:", best_acc)
    print("=" * 50)

#11.主函数入口与多进程支持
if __name__ == "__main__":

    multiprocessing.freeze_support()

    main()