import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

def plot_training_curves(train_losses, val_losses, val_accs, save_path):
    epochs = range(1, len(train_losses)+1)
    fig, ax1 = plt.subplots(figsize=(10,5))
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax2 = ax1.twinx()
    ax2.plot(epochs, val_accs, 'g--', label='Val Accuracy')
    ax2.set_ylabel('Accuracy')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labels1+labels2, loc='best')
    plt.title('Training Curves')
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()

def visualize_weights(W, classes, save_path, num_filters=16):
    # W: (input_dim, hidden_dim), 取前num_filters个神经元
    W = W[:, :num_filters]
    # 将每个神经元权重reshape为(64,64,3)
    fig, axes = plt.subplots(4, 4, figsize=(12,12))
    for i, ax in enumerate(axes.flat):
        if i < W.shape[1]:
            img = W[:, i].reshape(64, 64, 3)
            # 归一化到0-1显示
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f'Filter {i+1}')
    plt.suptitle('First Layer Weight Visualization')
    plt.savefig(save_path)
    plt.close()

def plot_misclassifications(X, y_true, y_pred, classes, num=9, save_path=None):
    errors = np.where(y_true != y_pred)[0]
    np.random.shuffle(errors)
    selected = errors[:num]
    fig, axes = plt.subplots(3, 3, figsize=(8,8))
    for idx, ax in zip(selected, axes.flat):
        img = X[idx].reshape(64,64,3)
        ax.imshow(img)
        ax.set_title(f'True: {classes[y_true[idx]]}\nPred: {classes[y_pred[idx]]}')
        ax.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()