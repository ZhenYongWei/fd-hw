from layers import *

class ThreeLayerMLP:
    def __init__(self, input_dim, hidden_dim, num_classes, activation='relu'):
        self.fc1 = Linear(input_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, hidden_dim)
        self.fc3 = Linear(hidden_dim, num_classes)

        if activation == 'relu':
            self.act1 = ReLU()
            self.act2 = ReLU()
        elif activation == 'sigmoid':
            self.act1 = Sigmoid()
            self.act2 = Sigmoid()
        elif activation == 'tanh':
            self.act1 = Tanh()
            self.act2 = Tanh()
        else:
            raise ValueError("Unsupported activation")

        self.loss_fn = SoftmaxCrossEntropyLoss()
        self.layers = [self.fc1, self.act1, self.fc2, self.act2, self.fc3]

    def forward(self, x):
        out = self.fc1.forward(x)
        out = self.act1.forward(out)
        out = self.fc2.forward(out)
        out = self.act2.forward(out)
        out = self.fc3.forward(out)
        return out

    def compute_loss(self, scores, y, l2_lambda=0.0):
        loss = self.loss_fn.forward(scores, y)
        if l2_lambda > 0:
            l2_loss = 0.5 * l2_lambda * (np.sum(self.fc1.W ** 2) +
                                         np.sum(self.fc2.W ** 2) +
                                         np.sum(self.fc3.W ** 2))
            loss += l2_loss
        return loss

    def backward(self, l2_lambda=0.0):
        dout = self.loss_fn.backward()
        # fc3 backward
        dout = self.fc3.backward(dout)
        if l2_lambda > 0:
            self.fc3.dW += l2_lambda * self.fc3.W
        # act2 backward
        dout = self.act2.backward(dout)
        # fc2 backward
        dout = self.fc2.backward(dout)
        if l2_lambda > 0:
            self.fc2.dW += l2_lambda * self.fc2.W
        # act1 backward
        dout = self.act1.backward(dout)
        # fc1 backward
        dout = self.fc1.backward(dout)
        if l2_lambda > 0:
            self.fc1.dW += l2_lambda * self.fc1.W

    def save_weights(self, path):
        if not path.endswith('.npz'):
            path = path + '.npz'
        np.savez(path, W1=self.fc1.W, b1=self.fc1.b,
                       W2=self.fc2.W, b2=self.fc2.b,
                       W3=self.fc3.W, b3=self.fc3.b)

    def load_weights(self, path):
        if not path.endswith('.npz'):
            path = path + '.npz'
        data = np.load(path)
        self.fc1.W = data['W1']
        self.fc1.b = data['b1']
        self.fc2.W = data['W2']
        self.fc2.b = data['b2']
        self.fc3.W = data['W3']
        self.fc3.b = data['b3']