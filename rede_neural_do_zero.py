import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from time import time
from torch import nn, optim


print("Numpy version:", np.__version__)
print("PyTorch version:", torch.__version__)

# Teste básico do PyTorch
x = torch.randn(2, 3)  # Cria um tensor aleatório
print("Tensor:", x)