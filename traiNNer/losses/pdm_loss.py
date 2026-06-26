import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812

from traiNNer.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class TVLoss(nn.Module):
    def __init__(self, loss_weight: float, penalty: str = "l1") -> None:
        super().__init__()
        self.loss_weight = loss_weight
        if penalty.lower() == "l2":
            self.penalty = nn.MSELoss()
        else:
            self.penalty = nn.L1Loss()

    def forward(self, pred: Tensor) -> Tensor:
        y_diff = self.penalty(pred[:, :, :-1, :], pred[:, :, 1:, :])
        x_diff = self.penalty(pred[:, :, :, :-1], pred[:, :, :, 1:])
        return x_diff + y_diff


@LOSS_REGISTRY.register()
class GaussGuidedLoss(nn.Module):
    def __init__(self, loss_weight: float, ksize: int, sigma: float) -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self.ksize = ksize

        ax = torch.arange(0, ksize, dtype=torch.float32) - ksize // 2
        yy, xx = torch.meshgrid(ax, ax, indexing="ij")
        dis = torch.exp(-(xx**2 + yy**2) / (sigma**2))
        dis = dis / dis.sum()
        self.register_buffer("gauss", dis.reshape(1, ksize * ksize, 1, 1))

    def forward(self, kernel: Tensor) -> Tensor:
        if kernel.ndim == 3 and kernel.shape[-1] == self.ksize:
            kernel = kernel.reshape(kernel.shape[0], self.ksize * self.ksize, 1, 1)
        elif kernel.ndim == 4 and kernel.shape[1] == self.ksize:
            kernel = kernel.reshape(
                kernel.shape[0], self.ksize * self.ksize, kernel.shape[2], kernel.shape[3]
            )
        elif kernel.ndim == 5:
            kernel = kernel.reshape(
                kernel.shape[0], self.ksize * self.ksize, kernel.shape[3], kernel.shape[4]
            )
        elif kernel.ndim != 4:
            raise ValueError(f"Unsupported kernel shape for GaussGuidedLoss: {tuple(kernel.shape)}")

        target = self.gauss.expand(kernel.shape[0], -1, kernel.shape[2], kernel.shape[3])
        return F.mse_loss(kernel, target)
