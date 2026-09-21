import math
import numpy as np
import torch
import torch.nn as nn


def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return float(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - (high - low) + low)


def grl_hook(coeff):
    def fun1(grad):
        return -coeff * grad.clone()
    return fun1


class DomainClassifier(nn.Module):
    def __init__(self):
        super(DomainClassifier, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.domain = nn.Linear(1024, 1)

    def forward(self, x, iter_num):
        coeff = calc_coeff(iter_num, 1.0, 0.0, 10, 10000.0)
        x.register_hook(grl_hook(coeff))
        x = self.layer(x)
        domain_y = self.domain(x)
        return domain_y


class RandomLayer(nn.Module):
    def __init__(self, input_dim_list=None, output_dim=1024):
        super().__init__()
        input_dim_list = input_dim_list or []
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim

        for i, in_dim in enumerate(input_dim_list):
            rm = torch.randn(in_dim, output_dim)
            self.register_buffer(f"random_matrix_{i}", rm)

    def forward(self, input_list):
        return_list = []
        for i in range(self.input_num):
            rm = getattr(self, f"random_matrix_{i}")
            return_list.append(torch.mm(input_list[i], rm))

        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0 / len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor
