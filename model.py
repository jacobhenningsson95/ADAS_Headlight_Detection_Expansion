import torch
from torch import nn
# same model from old project

Pool = nn.MaxPool2d


def batchnorm(x):
    return nn.BatchNorm2d(x.size()[1])(x)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        m.weight.data.normal_(0, (1./n)**0.5)


class Full(nn.Module):
    def __init__(self, inp_dim, out_dim, bn = False, relu = False):
        super(Full, self).__init__()
        self.fc = nn.Linear(inp_dim, out_dim, bias = True)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        x = self.fc(x.view(x.size()[0], -1))
        if self.relu is not None:
            x = self.relu(x)
        if self.bn is not None:
            x = self.bn(x)
        return x


class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride = 1, bn = False, relu = True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=True)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.relu is not None:
            x = self.relu(x)
        if self.bn is not None:
            x = self.bn(x)
        return x


class Hourglass(nn.Module):
    def __init__(self, n, f, bn=None, increase=128):
        super(Hourglass, self).__init__()
        nf = f + increase
        self.up1 = Conv(f, f, 3, bn=bn)
        # Lower branch
        self.pool1 = Pool(2, 2)
        self.low1 = Conv(f, nf, 3, bn=bn)
        # Recursive hourglass
        if n > 1:
            self.low2 = Hourglass(n-1, nf, bn=bn)
        else:
            self.low2 = Conv(nf, nf, 3, bn=bn)
        self.low3 = Conv(nf, f, 3)
        self.up2  = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, x):
        up1  = self.up1(x)
        pool1 = self.pool1(x)
        low1 = self.low1(pool1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2  = self.up2(low3)
        return up1 + up2


class Merge(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(Merge, self).__init__()
        self.conv = Conv(x_dim, y_dim, 1, relu=False, bn=False)

    def forward(self, x):
        return self.conv(x)


class PoseNet(nn.Module):
    def __init__(self, nstack, inp_dim, oup_dim, bn=False, increase=128, **kwargs):
        super(PoseNet, self).__init__()
        self.pre = nn.Sequential(
            Conv(1, 64, 7, 2, bn=bn),
            Conv(64, 128, bn=bn),
            Pool(2, 2),
            Conv(128, 128, bn=bn),
            Conv(128, inp_dim, bn=bn)
        )
        self.features = nn.ModuleList([
            nn.Sequential(
                Hourglass(4, inp_dim, bn, increase),
                Conv(inp_dim, inp_dim, 3, bn=False),
                Conv(inp_dim, inp_dim, 3, bn=False)
            ) for _ in range(nstack)])

        self.outs = nn.ModuleList([Conv(inp_dim, oup_dim, 1, relu=False, bn=False) for _ in range(nstack)])
        self.merge_features = nn.ModuleList([Merge(inp_dim, inp_dim) for _ in range(nstack-1)])
        self.merge_preds = nn.ModuleList([Merge(oup_dim, inp_dim) for _ in range(nstack-1)])
        self.nstack = nstack
        self.semantic_seg, self.instance_seg = \
            self.__generate_heads(64, 32)

    def __generate_heads(self, input_n_filters, embedding_size):

        semantic_segmentation = nn.Sequential()
        semantic_segmentation.add_module('Conv_1',
                                         nn.Conv2d(input_n_filters,
                                                   1,
                                                   kernel_size=(1, 1),
                                                   stride=(1, 1)))

        instance_segmentation = nn.Sequential()
        instance_segmentation.add_module('Conv_1',
                                         nn.Conv2d(input_n_filters,
                                                   embedding_size,
                                                   kernel_size=(1, 1),
                                                   stride=(1, 1)))

        return semantic_segmentation, instance_segmentation

    def forward(self, imgs):
        x = imgs.permute(0, 3, 1, 2)
        x = self.pre(x)
        preds = []
        for i in range(self.nstack):
            feature = self.features[i](x)
            preds.append(self.outs[i](feature))
            if i != self.nstack - 1:
                x = x + self.merge_preds[i](preds[-1]) + self.merge_features[i](feature)

        hourglass_output = torch.stack(preds, 1) # 16, 2, 64, 128, 128

        semantic_preds = []
        instance_preds = []

        for i in range(self.nstack):  # 16, 1, 64, 128, 128 -> 16, 64, 128, 128
            hourglass_dim = hourglass_output[:, i]

            semantic_preds.append(self.semantic_seg(hourglass_output[:, i] ))
            instance_preds.append(self.instance_seg(hourglass_output[:, i]))


        return [torch.stack(instance_preds, 1), torch.stack(semantic_preds, 1)]
