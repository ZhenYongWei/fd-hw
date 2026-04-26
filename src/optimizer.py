import numpy as np
from layers import *

class SGD:
    def __init__(self, layers, lr=0.01, decay=0.0, momentum=0.0):
        self.layers = layers  # 仅包含Linear层
        self.lr = lr
        self.initial_lr = lr
        self.decay = decay
        self.momentum = momentum
        self.velocities = None
        if momentum > 0:
            self.velocities = {}
            for i, layer in enumerate(layers):
                self.velocities[f'W{i}'] = np.zeros_like(layer.W)
                self.velocities[f'b{i}'] = np.zeros_like(layer.b)

    def step(self):
        linear_layers = [l for l in self.layers if isinstance(l, Linear)]
        for i, layer in enumerate(linear_layers):
            if self.momentum > 0:
                self.velocities[f'W{i}'] = self.momentum * self.velocities[f'W{i}'] + layer.dW
                self.velocities[f'b{i}'] = self.momentum * self.velocities[f'b{i}'] + layer.db
                layer.W -= self.lr * self.velocities[f'W{i}']
                layer.b -= self.lr * self.velocities[f'b{i}']
            else:
                layer.W -= self.lr * layer.dW
                layer.b -= self.lr * layer.db

    def schedule_lr(self, epoch):
        # 指数衰减
        self.lr = self.initial_lr * np.exp(-self.decay * epoch)