import torch
from torch import nn
from torch.nn.functional import pad
from torch.fft import rfft, irfft, rfft2, irfft2


class LTI(nn.Module):
    """
    State-Free Inference of State-Space Models:
    The Transfer Function Approach
    https://arxiv.org/abs/2405.06147
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        order: int = 8,
        causal: bool = True,
        mimo: bool = True,
        zero_init: bool = True,
    ):
        super().__init__()
        self.order = order
        self.causal = causal
        self.mimo = mimo
        assert mimo or (input_dim == output_dim)
        shape = (input_dim, output_dim) if mimo else (input_dim,)
        initializer = torch.zeros if zero_init else lambda s: torch.randn(s) / order
        self.h0 = nn.Parameter(initializer((*shape, 1)))
        self.numerators = nn.Parameter(initializer((2, *shape, order)))
        self.denumerator = nn.Parameter(initializer((*shape, order)))

    def __call__(self, x):
        *B, L, D = x.shape

        # compute truncated impulse response
        a = pad(self.denumerator, (1, 0), value=1.0)
        b = pad(self.numerators, (1, 0), value=0.0)
        H_truncated = rfft(b, n=L) / rfft(a, n=L)
        h_truncated = irfft(H_truncated, n=L)

        # transfer function padded to 2L for convolution
        H = rfft(h_truncated, n=2 * L)
        H_causal, H_anticausal = torch.unbind(H, dim=0)
        if self.causal:
            H = self.h0 + H_causal
        else:
            H = self.h0 + H_causal + H_anticausal.conj()

        # convolution in frequency domain
        X = rfft(x, n=2 * L, dim=-2)
        if self.mimo:
            Y = torch.einsum("...li,...ijl->...lj", X, H)
        else:
            Y = torch.einsum("...li,...il->...li", X, H)
        y = irfft(Y, dim=-2)
        y = y[..., :L, :]
        return y


class LTI2d(nn.Module):
    """
    State-Free Inference of State-Space Models:
    The Transfer Function Approach
    https://arxiv.org/abs/2405.06147
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        order: int = 8,
        causal: bool = True,
        mimo: bool = True,
        zero_init: bool = True,
    ):
        super().__init__()
        self.order = order
        self.causal = causal
        self.mimo = mimo
        assert mimo or (input_dim == output_dim)
        shape = (input_dim, output_dim) if mimo else (input_dim,)
        initializer = torch.zeros if zero_init else lambda s: torch.randn(s) / order
        self.h0 = nn.Parameter(initializer((*shape, 1, 1)))
        self.numerators = nn.Parameter(initializer((4, *shape, order, order)))
        self.denumerator = nn.Parameter(initializer((*shape, order, order)))

    def __call__(self, x):
        *B, L1, L2, D = x.shape
        # compute truncated impulse response
        a = pad(self.denumerator, (1, 0, 1, 0), value=1.0)
        b = pad(self.numerators, (1, 0, 1, 0), value=0.0)
        H_truncated = rfft2(b, s=(L1, L2)) / rfft2(a, s=(L1, L2))
        h_truncated = irfft2(H_truncated, s=(L1, L2))

        # transfer function padded to 2L for convolution
        H = rfft2(h_truncated, s=(2 * L1, 2 * L2))
        H_cc, H_ca, H_ac, H_aa = torch.unbind(H, dim=0)
        if self.causal:
            H = self.h0 + H_cc
        else:
            H_ca = torch.flip(H_ca, dims=(-2,))
            H_ac = torch.flip(H_ac, dims=(-2,)).conj()
            H_aa = H_aa.conj()
            H = self.h0 + H_cc + H_ca + H_ac + H_aa

        # convolution in frequency domain
        X = rfft2(x, s=(2 * L1, 2 * L2), dim=(-3, -2))
        if self.mimo:
            Y = torch.einsum("...hwi,...ijhw->...hwj", X, H)
        else:
            Y = torch.einsum("...hwi,...ihw->...hwi", X, H)
        y = irfft2(Y, s=(2 * L1, 2 * L2), dim=(-3, -2))
        y = y[..., :L1, :L2, :]
        return y
