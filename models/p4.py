
import torch
from torch import nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

eps = 1e-5

# adapted from github.com/claudio-unipv/groupcnn
class ConvZ2ToP4(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self._weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size).to(device))
        nn.init.kaiming_uniform_(self._weight, a=5 ** 0.5)
        self.bias = nn.Parameter(torch.zeros(out_channels).to(device)) if bias else None

        # make four copies of the filters, rotated element-wise
        self.weight = torch.stack([self._weight.rot90(k, dims=(2, 3)) for k in range(4)], dim=1)
        assert(self.weight.shape == (out_channels, 4, in_channels, kernel_size, kernel_size))

    # inp: n x c_in x h x w
    def forward(self, inp):
        n = inp.shape[0]
        # fold per-orientation filters into channel dims for convolution
        _weight = self.weight.view(self.out_channels * 4, self.in_channels, self.kernel_size, self.kernel_size)
        out = F.conv2d(inp, _weight, stride=self.stride, padding=self.padding)
        # extract per-orientation filters into their own dim
        out = out.view(n, self.out_channels, 4, out.shape[-2], out.shape[-1])
        if self.bias is not None:
            out += self.bias.view(self.out_channels, 1, 1, 1)
        return out

class ConvP4ToP4(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self._weight = nn.Parameter(torch.empty(out_channels, in_channels, 4, kernel_size, kernel_size).to(device))
        nn.init.kaiming_uniform_(self._weight, a=5 ** 0.5)
        self.bias = nn.Parameter(torch.zeros(out_channels).to(device)) if bias else None

        # make four copies of the filters, rotated element-wise AND around the feature map circle by 90deg each time
        self.weight = torch.stack([self._weight.rot90(k, dims=(3, 4)).roll(k, 2) for k in range(4)], dim=1)
        assert(self.weight.shape == (out_channels, 4, in_channels, 4, kernel_size, kernel_size))

    # inp: n x c_in x 4 x h x w
    def forward(self, inp):
        n = inp.shape[0]
        # fold per-orientation filters into channel dims for convolution
        _inp = inp.view(n, self.in_channels * 4, inp.shape[-2], inp.shape[-1])
        _weight = self.weight.view(self.out_channels * 4, self.in_channels * 4, self.kernel_size, self.kernel_size)
        out = F.conv2d(_inp, _weight, stride=self.stride, padding=self.padding)
        # extract per-orientation filters into their own dim
        out = out.view(n, self.out_channels, 4, out.shape[-2], out.shape[-1])
        if self.bias is not None:
            out += self.bias.view(self.out_channels, 1, 1, 1)
        return out

class InstanceNormP4(nn.Module):
    # are you supposed to do something with num_features?
    def __init__(self, num_features):
        super().__init__()

    # inp: n x c_in x 4 x h x w
    def forward(self, inp):
        mu = inp.mean(dim=(2, 3, 4), keepdim=True)
        sigma = ((inp - mu).square().mean(dim=(2, 3, 4), keepdim=True) + eps).sqrt()
        return (inp - mu) / sigma

class AvgPoolP4(nn.Module):
    # inp: n x c_in x 4 x h x w
    def forward(self, inp):
        return inp.mean(dim=2)

#net = nn.Sequential(
#    ConvZ2ToP4(3, 7, padding=2),
#    ConvP4ToP4(7, 7, padding=2),
#    ConvP4ToP4(7, 7, padding=2),
#    AvgPoolP4(),
#)
#
#inp1 = torch.randn(3, 12, 12)
#inp1r = inp1.rot90(k=1, dims=(1, 2))
#inputs = torch.stack([inp1, inp1r])
#
#out = net(inputs)
#(out[0] - out[1].rot90(k=-1, dims=(1, 2))).abs().max().item()
