import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

def load_data(data_dir, val_ratio=0.1, test_ratio=0.1, random_state=42):
    classes = sorted(os.listdir(data_dir))
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    X, y = [], []
    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        for fname in os.listdir(cls_dir):
            img = Image.open(os.path.join(cls_dir, fname)).convert('RGB')
            img = img.resize((64, 64))  # 确保尺寸统一
            arr = np.array(img, dtype=np.float32) / 255.0  # 归一化到[0,1]
            X.append(arr.flatten())
            y.append(class_to_idx[cls])
    X = np.array(X)
    y = np.array(y, dtype=np.int32)
    # 划分训练/验证/测试集
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(val_ratio+test_ratio),
                                                        random_state=random_state, stratify=y)
    val_size = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1-val_size,
                                                    random_state=random_state, stratify=y_temp)
    return X_train, y_train, X_val, y_val, X_test, y_test, classes