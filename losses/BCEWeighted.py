import torch
import torch.nn as nn


class BCEWeighted(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        assert reduction in ["none", "mean"]
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.reduction = reduction

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        seq_len = output.size()[1]
        weights = torch.clone(target)
        weights[weights == 0] = 1.1 / seq_len

        loss = weights * self.bce(output, target)
        if self.reduction == "mean":
            loss = torch.mean(loss)
        return loss
