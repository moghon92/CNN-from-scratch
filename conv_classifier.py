
from .softmax_ce import SoftmaxCrossEntropy
from .relu import ReLU
from .max_pool import MaxPooling
from .convolution import Conv2D
from .linear import Linear


class ConvNet:
    """
    Max Pooling of input
    """
    def __init__(self, modules, criterion):
        self.modules = []
        for m in modules:
            if m['type'] == 'Conv2D':
                self.modules.append(
                    Conv2D(m['in_channels'],
                           m['out_channels'],
                           m['kernel_size'],
                           m['stride'],
                           m['padding'])
                )
            elif m['type'] == 'ReLU':
                self.modules.append(
                    ReLU()
                )
            elif m['type'] == 'MaxPooling':
                self.modules.append(
                    MaxPooling(m['kernel_size'],
                               m['stride'])
                )
            elif m['type'] == 'Linear':
                self.modules.append(
                    Linear(m['in_dim'],
                           m['out_dim'])
                )
        if criterion['type'] == 'SoftmaxCrossEntropy':
            self.criterion = SoftmaxCrossEntropy()
        else:
            raise ValueError("Wrong Criterion Passed")

    def forward(self, x, y):
        """
        The forward pass of the model
        :param x: input data: (N, C, H, W)
        :param y: input label: (N, )
        :return:
          probs: the probabilities of all classes: (N, num_classes)
          loss: the cross entropy loss
        """
        probs = None
        loss = None

        layer_in = x
        for m in self.modules:
            layer_out = m.forward(layer_in)
            layer_in = layer_out

        probs, loss = self.criterion.forward(layer_out, y)

        return probs, loss

    def backward(self):
        """
        The backward pass of the model
        :return: nothing but dx, dw, and db of all modules are updated
        """

        self.criterion.backward()
        upstream_grad = self.criterion.dx
        for m in self.modules[::-1]:
            m.backward(upstream_grad)
            upstream_grad = m.dx
