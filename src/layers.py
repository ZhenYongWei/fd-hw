import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        # He初始化
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.b = np.zeros((1, out_features))
        self.x = None  # 缓存输入，用于反向传播

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout):
        # dout: loss对输出的梯度 (N, out_features)
        self.dW = self.x.T @ dout
        self.db = np.sum(dout, axis=0, keepdims=True)
        dx = dout @ self.W.T
        return dx

class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x > 0)
        return x * self.mask

    def backward(self, dout):
        return dout * self.mask

class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out

    def backward(self, dout):
        return dout * self.out * (1 - self.out)

class Tanh:
    def __init__(self):
        self.out = None

    def forward(self, x):
        self.out = np.tanh(x)
        return self.out

    def backward(self, dout):
        return dout * (1 - self.out ** 2)

class SoftmaxCrossEntropyLoss:
    def __init__(self):
        self.probs = None
        self.labels = None

    def forward(self, scores, y):
        # scores: (N, C), y: (N,) 整数标签
        shifted = scores - np.max(scores, axis=1, keepdims=True)
        exps = np.exp(shifted)
        self.probs = exps / np.sum(exps, axis=1, keepdims=True)
        self.labels = y
        N = scores.shape[0]
        loss = -np.log(self.probs[np.arange(N), y]).mean()
        return loss

    def backward(self):
        N = self.labels.shape[0]
        dout = self.probs.copy()
        dout[np.arange(N), self.labels] -= 1
        dout /= N
        return dout