

import numpy as np


class MaxPooling:
    """
    Max Pooling of input
    """

    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride
        self.cache = None
        self.dx = None

    def forward(self, x):
        """
        Forward pass of max pooling
        :param x: input, (N, C, H, W)
        :return: The output by max pooling with kernel_size and stride
        """
        out = None
        k = self.kernel_size
        s = self.stride

        N = x.shape[0]
        c = x.shape[1]
        H_out = ((x.shape[2] - k) // s) + 1
        W_out = ((x.shape[3] - k) // s) + 1

        x_views = np.lib.stride_tricks.as_strided(x,
                                                  shape=(N, c, H_out, W_out, k, k),
                                                  strides=(x.strides[0], x.strides[1], x.strides[2] * s,
                                                           x.strides[3] * s, x.strides[2], x.strides[3]),
                                                  writeable=False)

        out = np.max(x_views, axis=(4, 5))

        self.cache = (x, H_out, W_out)
        return out

    def backward(self, dout):
        """
        Backward pass of max pooling
        :param dout: Upstream derivatives
        :return:
        """
        x, H_out, W_out = self.cache

       # print(H_out, W_out)

        k = self.kernel_size
        s = self.stride

        N = x.shape[0]
        c = x.shape[1]

        x_views = np.lib.stride_tricks.as_strided(x,
                                                  shape=(N, c, H_out, W_out, k, k),
                                                  strides=(x.strides[0], x.strides[1], x.strides[2] * s,
                                                           x.strides[3] * s, x.strides[2], x.strides[3]),
                                                  writeable=False)

        x_view_reshape = x_views.reshape((N, c, H_out, W_out, -1))
        indexes = np.argmax(x_view_reshape, axis=-1)

       # print(x_views)
        dx_views = np.zeros(x_views.shape)
        for i in range(indexes.shape[0]):
            for j in range(indexes.shape[1]):
                for n in range(indexes.shape[2]):
                    for q in range(indexes.shape[3]):
                        index = indexes[i, j, n, q]
                        index2 = np.unravel_index(index, x_views[i, j, n, q, :, :].shape)
                #        print(index2)
                        dx_views[i, j, n, q, index2[0], index2[1]] = 1



        #print(dx_views)
        gradients = np.einsum('NchwKk,Nchw->NchwKk', dx_views, dout)
        #print(gradients)
        dx = np.zeros(x.shape)

        s1 = 0
        for h in range(0, H_out):
            s2 = 0
            for w in range(0, W_out):
                dx[:, :, s1:s1 + k, s2:s2 + k] += gradients[:, :, h, w, :, :]
                s2 += s
                if s2 > x.shape[3]:
                    break
            s1 += s
            if s2 > x.shape[2]:
                break

        #for h in range(0,H_out,s):
        #    for w in range(0,W_out,s):
        #        dx[:, :, h:h+k, w:w+k] += gradients[:, :, h, w, :, :]

        #print(dx)
        self.dx = dx



