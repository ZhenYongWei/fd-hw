import numpy as np
import os
import argparse
from data_loader import load_data
from model import ThreeLayerMLP
from optimizer import SGD
from layers import Linear
from utils import plot_training_curves, visualize_weights

def train(args):
    # 加载数据
    X_train, y_train, X_val, y_val, _, _, classes = load_data(args.data_dir)
    input_dim = X_train.shape[1]
    num_classes = len(classes)
    model = ThreeLayerMLP(input_dim, args.hidden_dim, num_classes, args.activation)
    linear_layers = [model.fc1, model.fc2, model.fc3]
    optimizer = SGD(linear_layers, lr=args.lr, decay=args.lr_decay, momentum=args.momentum)

    best_val_acc = 0
    train_losses, val_losses, val_accs = [], [], []

    for epoch in range(args.epochs):
        # 打乱训练数据
        perm = np.random.permutation(X_train.shape[0])
        X_shuffled = X_train[perm]
        y_shuffled = y_train[perm]

        # Mini-batch训练
        total_loss = 0
        num_batches = 0
        for i in range(0, X_train.shape[0], args.batch_size):
            X_batch = X_shuffled[i:i+args.batch_size]
            y_batch = y_shuffled[i:i+args.batch_size]
            scores = model.forward(X_batch)
            loss = model.compute_loss(scores, y_batch, args.l2_lambda)
            model.backward(args.l2_lambda)
            optimizer.step()
            total_loss += loss
            num_batches += 1

        train_loss = total_loss / num_batches
        train_losses.append(train_loss)

        # 验证集评估
        val_scores = model.forward(X_val)
        val_loss = model.compute_loss(val_scores, y_val, args.l2_lambda)
        val_preds = np.argmax(val_scores, axis=1)
        val_acc = np.mean(val_preds == y_val)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_weights(args.save_path)
            print(f"  -> New best model saved with accuracy {best_val_acc:.4f}")

        # 学习率衰减
        optimizer.schedule_lr(epoch)

    # 绘制曲线
    plot_training_curves(train_losses, val_losses, val_accs, os.path.join(args.output_dir, 'training_curves.png'))
    # 权重可视化
    best_model = ThreeLayerMLP(input_dim, args.hidden_dim, num_classes, args.activation)
    best_model.load_weights(args.save_path)
    visualize_weights(best_model.fc1.W, classes, os.path.join(args.output_dir, 'weight_vis.png'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/EuroSAT_RGB')
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr_decay', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--l2_lambda', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--save_path', type=str, default='./results/best_model')
    parser.add_argument('--output_dir', type=str, default='./results')
    args = parser.parse_args()
    train(args)