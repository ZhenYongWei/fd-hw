import numpy as np
import argparse
from data_loader import load_data
from model import ThreeLayerMLP
from utils import plot_confusion_matrix, plot_misclassifications

def test(args):
    _, _, _, _, X_test, y_test, classes = load_data(args.data_dir)
    model = ThreeLayerMLP(X_test.shape[1], args.hidden_dim, len(classes), args.activation)
    model.load_weights(args.model_path)
    scores = model.forward(X_test)
    preds = np.argmax(scores, axis=1)
    acc = np.mean(preds == y_test)
    print(f"Test Accuracy: {acc:.4f}")
    plot_confusion_matrix(y_test, preds, classes, args.confusion_matrix_path)
    plot_misclassifications(X_test, y_test, preds, classes, save_path=args.error_analysis_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data/EuroSAT_RGB')
    parser.add_argument('--model_path', type=str, default='../results/best_model.npy')
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--confusion_matrix_path', type=str, default='../results/confusion_matrix.png')
    parser.add_argument('--error_analysis_path', type=str, default='../results/error_analysis.png')
    args = parser.parse_args()
    test(args)