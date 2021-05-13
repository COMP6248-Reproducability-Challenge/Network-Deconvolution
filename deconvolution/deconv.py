import torch
from torch import nn
import torch.nn.functional as F

from torch.nn.modules import conv
from torch.nn.modules.utils import _pair
import math

def isqrt_newton_schulz_autograd(A, numIters):
    dim = A.shape[0]
    normA = A.norm()
    Y = A.div(normA)
    I = torch.eye(dim, dtype=A.dtype, device=A.device)
    Z = torch.eye(dim, dtype=A.dtype, device=A.device)

    for i in range(numIters):
        T = 0.5 * (3.0 * I - Z @ Y)
        Y = Y @ T
        Z = T @ Z
    # A_sqrt = Y * torch.sqrt(normA)
    A_isqrt = Z / torch.sqrt(normA)
    return A_isqrt


class FastDeconv(conv._ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 eps=1e-5, n_iter=5, momentum=0.1, block=64, sampling_stride=3, freeze=False, freeze_iter=100):
        super(FastDeconv, self).__init__(
            in_channels, out_channels, _pair(kernel_size), _pair(stride), _pair(padding), _pair(dilation),
            False, _pair(0), 1, bias, padding_mode='zeros')

        self.momentum = momentum
        self.n_iter = n_iter
        self.eps = eps
        self.sampling_stride = sampling_stride * self.stride[0]
        self.counter = 0

        if block > in_channels:
            block = in_channels
        else:
            if in_channels % block != 0:
                block = math.gcd(block, in_channels)
        self.block = block
        self.num_features = self.kernel_size[0]**2 * block

        self.register_buffer('running_mean', torch.zeros(self.num_features))
        self.register_buffer('running_deconv', torch.eye(self.num_features))

        self.freeze_iter = freeze_iter
        self.freeze = freeze

    def forward(self, x):
        N, C, H, W = x.shape
        B = self.block

        frozen = self.freeze and (self.counter > self.freeze_iter)
        if self.training:
            self.counter += 1
            self.counter %= (self.freeze_iter * 10)

        if self.training and not frozen:
            # 1. im2col
            # (batch, n_samples, n_features = chin * kernel**2)
            X = torch.nn.functional.unfold(x, self.kernel_size, self.dilation, self.padding,
                                           self.sampling_stride).transpose(1, 2).contiguous()
            # (batch * n_samples * n_blocks, n_features / n_blocks), n_blocks = chin / block_size
            X = X.view(-1, self.num_features, C // B).transpose(1, 2).contiguous().view(-1, self.num_features)

            # 2. subtract mean
            X_mean = X.mean(dim=0)
            X = X - X_mean

            # 3. calculate COV, COV^(-0.5), then deconv
            Id = torch.eye(X.shape[1], dtype=X.dtype, device=X.device)
            Cov = torch.addmm(Id, X.t(), X, beta=self.eps, alpha=1. / X.shape[0])
            deconv = isqrt_newton_schulz_autograd(Cov, self.n_iter)

            self.running_mean.mul_(1 - self.momentum)
            self.running_mean.add_(X_mean.detach() * self.momentum)
            self.running_deconv.mul_(1 - self.momentum)
            self.running_deconv.add_(deconv.detach() * self.momentum)

        else:
            X_mean = self.running_mean
            deconv = self.running_deconv

        # 4. X * deconv * conv = X * (deconv * conv)
        w = self.weight.view(-1, self.num_features, C // B).transpose(1, 2).contiguous().view(-1,
                                                                                              self.num_features) @ deconv
        b = self.bias - (w @ (X_mean.unsqueeze(1))).view(self.weight.shape[0], -1).sum(1)
        w = w.view(-1, C // B, self.num_features).transpose(1, 2).contiguous()
        w = w.view(self.weight.shape)
        x = F.conv2d(x, w, b, self.stride, self.padding, self.dilation, self.groups)
        return x


class DeLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, eps=1e-5, n_iter=5, momentum=0.1, block=512):
        super(DeLinear, self).__init__(in_features, out_features, bias)
        if block > in_features:
            block = in_features
        else:
            if in_features % block != 0:
                block = math.gcd(block, in_features)
                print('block size set to:', block)
        self.block = block
        self.momentum = momentum
        self.n_iter = n_iter
        self.eps = eps
        self.register_buffer('running_mean', torch.zeros(self.block))
        self.register_buffer('running_deconv', torch.eye(self.block))

    def forward(self, input):
        if self.training:
            # 1. reshape
            X = input.view(-1, self.block)

            # 2. subtract mean
            X_mean = X.mean(dim=0)
            X = X - X_mean
            self.running_mean.mul_(1 - self.momentum)
            self.running_mean.add_(X_mean.detach() * self.momentum)

            # 3. calculate COV, COV^(-0.5), then deconv
            Id = torch.eye(X.shape[1], dtype=X.dtype, device=X.device)
            Cov = torch.addmm(Id, X.t(), X, beta=self.eps, alpha=1. / X.shape[0])
            deconv = isqrt_newton_schulz_autograd(Cov, self.n_iter)

            self.running_mean.mul_(1 - self.momentum)
            self.running_mean.add_(X_mean.detach() * self.momentum)
            self.running_deconv.mul_(1 - self.momentum)
            self.running_deconv.add_(deconv.detach() * self.momentum)

        else:
            X_mean = self.running_mean
            deconv = self.running_deconv

        # 4. X * deconv * w = X * (deconv * w)
        w = self.weight.view(-1, self.block) @ deconv
        if self.bias is None:
            b = - (w @ (X_mean.unsqueeze(1))).view(self.weight.shape[0], -1).sum(1)
        else:
            b = self.bias - (w @ (X_mean.unsqueeze(1))).view(self.weight.shape[0], -1).sum(1)
        w = w.view(self.weight.shape)

        return F.linear(input, w, b)