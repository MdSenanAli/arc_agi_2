import torch
import torch.nn as nn
import torch.nn.functional as F


class SPPLayer(nn.Module):
    def __init__(self, levels=[1, 2, 4], pool_types=["max", "avg", "min"]):
        super().__init__()
        self.levels = levels
        self.pool_types = pool_types

    def forward(self, x):
        x = x.unsqueeze(0).unsqueeze(0)
        B, _, _, _ = x.size()
        spp_output = []

        for level in self.levels:
            for pool_type in self.pool_types:
                if pool_type == "max":
                    tensor = F.adaptive_max_pool2d(x, output_size=(level, level))
                elif pool_type == "avg":
                    tensor = F.adaptive_avg_pool2d(x, output_size=(level, level))
                elif pool_type == "min":
                    tensor = -F.adaptive_max_pool2d(-x, output_size=(level, level))

                spp_output.append(tensor.view(B, -1))  # Flatten each grid level

        return torch.cat(spp_output, dim=1)
