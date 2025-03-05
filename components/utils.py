import torch.nn as nn


def weight_init(m: nn.Module):
    for p in m.parameters():
        if isinstance(p, nn.Linear):
            nn.init.xavier_uniform_(p.weight.data)
            nn.init.constant_(p.bias.data, 0)
        elif isinstance(p, nn.Conv2d):
            nn.init.xavier_uniform_(p.weight.data)
            nn.init.constant_(p.bias.data, 0)

def soft_update_params(net: nn.Module, target_net: nn.Module, tau: float):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.lerp_(param.data, tau)