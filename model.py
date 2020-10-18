import torch
import torch.nn as nn
import torch.nn.functional as F


class LambdaConv(nn.Module):
    def __init__(self, in_channels, out_channels, heads=4, k=16, u=1, m=23):
        super(LambdaConv, self).__init__()
        self.kk, self.uu, self.vv, self.mm, self.heads = k, u, out_channels, m, heads
        self.local_context = True if m > 0 else False
        self.padding = (m - 1) // 2

        self.queries = nn.ModuleList([nn.Sequential(
            nn.Conv2d(in_channels, k // heads, kernel_size=1, bias=False),
            nn.BatchNorm2d(k // heads),
        ) for _ in range(self.heads)])
        self.keys = nn.ModuleList([nn.Sequential(
            nn.Conv2d(in_channels, k * u // heads, kernel_size=1, bias=False),
            nn.BatchNorm2d(k * u // heads),
        ) for _ in range(self.heads)])
        self.values = nn.ModuleList([nn.Conv2d(in_channels, self.vv * u // self.heads, kernel_size=1, bias=False) for _ in range(self.heads)])

        self.softmax = nn.Softmax(dim=1)

        if self.local_context:
            self.embedding = nn.Parameter(torch.randn([self.kk // self.heads, self.heads, self.uu, 1, m, m]), requires_grad=True)
        else:
            self.embedding = nn.Parameter(torch.randn([1, 1, 1, self.kk // self.heads, self.heads, self.uu]), requires_grad=True)

    def forward(self, x):
        n_batch, C, w, h = x.size()

        queries = [self.queries[_](x).permute(0, 2, 3, 1).unsqueeze(dim=-1) for _ in range(self.heads)]
        softmax = [self.softmax(self.keys[_](x).permute(0, 2, 3, 1).view(n_batch, -1, self.kk * self.uu // self.heads)).view(n_batch, w, h, -1, self.uu) for _ in range(self.heads)]
        values = [self.values[_](x).permute(0, 2, 3, 1).view(n_batch, w, h, self.vv // self.heads, self.uu) for _ in range(self.heads)]

        lambda_c = [torch.matmul(softmax[_], values[_].transpose(3, 4)).sum(dim=[1, 2], keepdim=True) for _ in range(self.heads)]
        y_c = [torch.matmul(queries[_].transpose(3, 4), lambda_c[_]) for _ in range(self.heads)]
        y_c = torch.cat(y_c, dim=3)

        if self.local_context:
            lambda_bnvk = [F.conv3d(values[_].permute(0, 4, 3, 1, 2), self.embedding[:, _, :, :, :, :], padding=(0, self.padding, self.padding)) for _ in range(self.heads)]
            lambda_p = [lambda_bnvk[_].permute(0, 3, 4, 1, 2).sum(dim=[1, 2], keepdim=True) for _ in range(self.heads)]
        else:
            lambda_p = [torch.matmul(self.embedding[:, :, :, :, _, :], values[_].transpose(3, 4)).sum(dim=[1, 2], keepdim=True) for _ in range(self.heads)]
        y_p = [torch.matmul(queries[_].transpose(3, 4), lambda_p[_]) for _ in range(self.heads)]
        y_p = torch.cat(y_p, dim=3)

        out = y_c + y_p
        out = out.view(n_batch, w, h, -1).permute(0, 3, 1, 2)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = LambdaConv(in_planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = LambdaConv(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = LambdaConv(planes, self.expansion * planes)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                LambdaConv(in_planes, self.expansion*planes),
                nn.BatchNorm2d(self.expansion*planes)
            )

        self.down = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.down = nn.Sequential(
                nn.AvgPool2d(kernel_size=(3, 3), stride=stride, padding=(1, 1))
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        out = self.down(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def LambdaResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def LambdaResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


# reference
# https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325
def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


model = LambdaResNet50()
print(get_n_params(model))

model = LambdaResNet152()
print(get_n_params(model))