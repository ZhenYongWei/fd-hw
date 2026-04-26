import itertools
import numpy as np
from data_loader import load_data
from model import ThreeLayerMLP
from optimizer import SGD
from layers import Linear

def run_search(data_dir):
    X_train, y_train, X_val, y_val, _, _, classes = load_data(data_dir)
    input_dim = X_train.shape[1]
    num_classes = len(classes)

    # 网格搜索空间
    hidden_dims = [128, 256, 512]
    lrs = [0.001, 0.01, 0.1]
    l2_lambdas = [0.0, 0.001, 0.01]
    activations = ['relu', 'tanh']

    results = []
    best_acc = 0
    best_config = None

    for hd, lr, l2, act in itertools.product(hidden_dims, lrs, l2_lambdas, activations):
        model = ThreeLayerMLP(input_dim, hd, num_classes, act)
        linear_layers = [model.fc1, model.fc2, model.fc3]
        opt = SGD(linear_layers, lr=lr, decay=0.0, momentum=0.9)
        # 简单训练20个epoch进行评估
        for epoch in range(20):
            perm = np.random.permutation(X_train.shape[0])
            for i in range(0, X_train.shape[0], 64):
                X_batch = X_train[perm[i:i+64]]
                y_batch = y_train[perm[i:i+64]]
                scores = model.forward(X_batch)
                loss = model.compute_loss(scores, y_batch, l2)
                model.backward(l2)
                opt.step()
        val_scores = model.forward(X_val)
        val_acc = np.mean(np.argmax(val_scores, axis=1) == y_val)
        results.append((hd, lr, l2, act, val_acc))
        print(f"Hidden={hd}, LR={lr}, L2={l2}, Act={act} -> Val Acc: {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            best_config = (hd, lr, l2, act)

    print(f"\nBest config: hidden_dim={best_config[0]}, lr={best_config[1]}, l2_lambda={best_config[2]}, activation={best_config[3]} with accuracy {best_acc:.4f}")
    # 保存搜索结果
    np.save('../results/hyper_search.npy', results)

if __name__ == '__main__':
    run_search('../data/EuroSAT_RGB')