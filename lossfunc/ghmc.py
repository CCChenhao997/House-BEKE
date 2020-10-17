import torch
import torch.nn as nn


class GHMC(nn.Module):
    def __init__(self, bins=10):
        super().__init__()
        self.bins = bins

    def forward(self, pred, target):
        target = target.squeeze(-1)
        device = target.device
        pred = nn.Sigmoid()(pred).flatten()
        edges = torch.arange(self.bins + 1).float().to(device) / self.bins  #定义区间边界
        g = torch.abs(pred.detach() - target)  #输出预测难度
        tot = pred.size(0)        # 计算所有有效样本总数
        n = 0  # n valid bins
        # 通过循环计算落入10个bins的梯度模长数量
        weights = torch.zeros_like(pred).to(device)
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i + 1])
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                # 重点，所谓的梯度密度就是1/num_in_bin
                weights[inds] = tot / num_in_bin
                n += 1
        if n > 0:
            weights = weights / n
        criterion = nn.BCELoss(weight=weights)
        loss = criterion(pred, target.float())
        return loss