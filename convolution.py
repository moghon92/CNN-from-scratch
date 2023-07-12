import numpy as np


class Conv2D:
    '''
    An implementation of the convolutional layer. We convolve the input with out_channels different filters
    and each filter spans all channels in the input.
    '''

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        """
        :param in_channels: the number of channels of the input data
        :param out_channels: the number of channels of the output(aka the number of filters applied in the layer)
        :param kernel_size: the specified size of the kernel(both height and width)
        :param stride: the stride of convolution
        :param padding: the size of padding. Pad zeros to the input with padding size.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.cache = None

        self._init_weights()

    def _init_weights(self):
        np.random.seed(1024)
        self.weight = 1e-3 * np.random.randn(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        self.bias = np.zeros(self.out_channels)

        self.dx = None
        self.dw = None
        self.db = None

    def forward(self, x):
        """
        The forward pass of convolution
        :param x: input data of shape (N, C, H, W)
        :return: output data of shape (N, self.out_channels, H', W') where H' and W' are determined by the convolution
                 parameters. Save necessary variables in self.cache for backward pass
        """
        out = None

        ############# fully vectorized! :) #############

        k = self.kernel_size
        s = self.stride
        p = self.padding

        x_pad = np.pad(x, [(0, 0), (0, 0), (p, p), (p, p)])

        N = x_pad.shape[0]
        c = self.in_channels
        h_out = ((x_pad.shape[2] - k) // s) + 1
        w_out = ((x_pad.shape[3] - k) // s) + 1

        x_views = np.lib.stride_tricks.as_strided(x_pad,
                                                shape=(N, h_out, w_out, c, k, k),
                                                strides=(x_pad.strides[0], x_pad.strides[2] * s, x_pad.strides[3] * s,
                                                         x_pad.strides[1], x_pad.strides[2], x_pad.strides[3]),
                                                writeable=False)

        out = np.einsum('NhwcKk,ocKk->Nohw', x_views, self.weight) + self.bias[:, np.newaxis, np.newaxis]

        #j = h_out * w_out
        #views_flat = x_views.reshape((N, j, -1))

        #l = self.out_channels
        #weight_flat = self.weight.reshape((l, -1)).T

        #out = np.einsum('ijk,kl->ilj', views_flat, weight_flat) + self.bias[:, np.newaxis]
        #out = out.reshape(N, self.out_channels, h_out, w_out)

        self.cache = x
        return out

    def backward(self, dout):
        """
        The backward pass of convolution
        :param dout: upstream gradients
        :return: nothing but dx, dw, and db of self should be updated
        """
        x = self.cache

        k = self.kernel_size
        s = self.stride
        p = self.padding

        x_pad = np.pad(x, [(0, 0), (0, 0), (p, p), (p, p)])

        N = x_pad.shape[0]
        c = self.in_channels
        h_out = ((x_pad.shape[2] - k) // s) + 1
        w_out = ((x_pad.shape[3] - k) // s) + 1

        x_views = np.lib.stride_tricks.as_strided(x_pad,
                                                shape=(N, h_out, w_out, c, k, k),
                                                strides=(x_pad.strides[0], x_pad.strides[2] * s, x_pad.strides[3] * s,
                                                         x_pad.strides[1], x_pad.strides[2], x_pad.strides[3]),
                                                writeable=False)

        self.dw = np.einsum('NhwcKk,Nohw->ocKk', x_views, dout)



        # j = h_out * w_out
        # views_flat = x_views.reshape((N, j, -1))

        # l = self.out_channels
        # dout_flat = dout.reshape((N, l, -1))

        # dw = np.einsum('ijk,ilj->lk', views_flat, dout_flat)
        # dw = dw.reshape(self.out_channels, c, k, k)
        #self.dw = dw


       # W = np.rot90(self.weight, 2,(2,3))
        w_views = np.lib.stride_tricks.as_strided(self.weight,
                                                shape=(h_out, w_out, self.out_channels, c, k, k),
                                                strides=(0, 0, self.weight.strides[0], self.weight.strides[1],
                                                         self.weight.strides[2], self.weight.strides[3]),
                                                writeable=False)

      #   p_h = (x.shape[2] - h_out) // 2
      #   p_w = (x.shape[3] - w_out) // 2
      #  dout_pad = np.pad(dout, [(0, 0), (0, 0), (p_h, p_h), (p_w, p_w)])
      # self.dx = self.dx[:, :, p_h:-p_h, p_w:-p_w]
        ##self.dx = np.einsum('hwocKk,Nohw->Nchw', w_views, dout)

        gradients = np.einsum('hwocKk,Nohw->NhwcKk', w_views, dout)
        out_pad = np.zeros(x_pad.shape)

        s1 = 0
        for h in range(h_out):
            s2 = 0
            for w in range(w_out):
                out_pad[:, :, s1:s1 + k, s2:s2 + k] += gradients[:, h, w, :, :, :]
                s2 += s
                if s2 > x.shape[3]:
                    break
            s1 += s
            if s2 > x.shape[2]:
                break

        #for h in range(h_out):
        #    for w in range(w_out):
        #        out_pad[:, :, h:h+k, w:w+k] += gradients[:, h, w, :, :, :]

        self.dx = out_pad[:, :, p:-p, p:-p]
        self.db = np.einsum('Nohw->o', dout)

