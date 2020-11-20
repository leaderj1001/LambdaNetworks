# LambdaNetworks: Modeling long-range Interactions without Attention

## Experimnets (CIFAR10)

| Model | k | h | u | m | Params (M) | Acc (%) |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| ResNet18 baseline ([ref](https://github.com/kuangliu/pytorch-cifar)) ||||| 14 | 93.02
| LambdaResNet18 | 16 | 4 | 4 | 9 | 8.6 | 92.21 (70 Epochs) |
| LambdaResNet18 | 16 | 4 | 4 | 7 | 8.6 | 94.20 (67 Epochs) |
| LambdaResNet18 | 16 | 4 | 4 | 5 | 8.6 | 91.58 (70 Epochs) |
| LambdaResNet18 | 16 | 4 | 1 | 23 | 8 | 91.36 (69 Epochs) |
| ResNet50 baseline ([ref](https://github.com/kuangliu/pytorch-cifar)) ||||| 23.5 | 93.62 |
| LambdaResNet50 | 16 | 4 | 4 | 7 | 13 | 93.74 (70 epochs) |

## Usage
```python
import torch

from model import LambdaConv, LambdaResNet50, LambdaResNet152

x = torch.randn([2, 3, 32, 32])
conv = LambdaConv(3, 128)
print(conv(x).size()) # [2, 128, 32, 32]

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
print(get_n_params(model)) # 14.9M (Ours) / 15M(Paper)

model = LambdaResNet152()
print(get_n_params(model)) # 32.8M (Ours) / 35M (Paper)
```

## Parameters
| Model | k | h | u | m | Params (M), Paper | Params (M), Ours |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|LambdaResNet50| 16 | 4 | 1 | 23 | 15.0 | 14.9 |
|LambdaResNet50| 16 | 4 | 4 | 7 | 16.0 | 16.0 |
|LambdaResNet152| 16 | 4 | 1 | 23 | 35 | 32.8 |
|LambdaResNet200| 16 | 4 | 1 | 23 | 42 | 35.29 |

## Ablation Parameters
| k | h | u | Params (M), Paper | Params (M), Ours |
|:-:|:-:|:-:|:-:|:-:|
| ResNet baseline ||| 25.6 | 25.5
| 8 | 2 | 1 | 14.8 | 15.0 |
| 8 | 16 | 1 | 15.6 | 14.9 |
| 2 | 4 | 1 | 14.7 | 14.6 |
| 4 | 4 | 1 | 14.7 | 14.66 |
| 8 | 4 | 1 | 14.8 | 14.66 |
| 16 | 4 | 1 | 15.0 | 14.99 |
| 32 | 4 | 1 | 15.4 | 15.4 |
| 2 | 8 | 1 | 14.7 | 14.5 |
| 4 | 8 | 1 | 14.7 | 14.57 |
| 8 | 8 | 1 | 14.7 | 14.74 |
| 16 | 8 | 1 | 15.1 | 14.1 |
| 32 | 8 | 1 | 15.7 | 15.76 |
| 8 | 8 | 4 | 15.3 | 15.26 |
| 8 | 8 | 8 | 16.0 | 16.0 |
| 16 | 4 | 4 | 16.0 | 16.0 |
