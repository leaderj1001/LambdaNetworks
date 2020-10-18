# LambdaNetworks

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
print(get_n_params(model)) # 16.6M / 15M(Paper)

model = LambdaResNet152()
print(get_n_params(model)) # 37.4M (Ours) / 35M (Paper)
```
