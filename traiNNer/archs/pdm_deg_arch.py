import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812

from traiNNer.utils.registry import ARCH_REGISTRY


class PDMResBlock(nn.Module):
    def __init__(self, nf: int, ksize: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(nf, nf, ksize, 1, ksize // 2),
            nn.BatchNorm2d(nf),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, ksize, 1, ksize // 2),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.body(x)


class PDMFourierQuantization(nn.Module):
    def __init__(self, n: int = 5) -> None:
        super().__init__()
        self.n = n

    def forward(self, inp: Tensor) -> Tensor:
        out = inp * 255.0
        flag = -1.0
        for i in range(1, self.n + 1):
            out = out + flag / np.pi / i * torch.sin(2 * i * np.pi * inp * 255.0)
            flag *= -1.0
        return out / 255.0


class PDMKernelModel(nn.Module):
    def __init__(self, opt: dict, scale: int) -> None:
        super().__init__()
        self.opt = opt
        self.scale = scale

        nc = int(opt["nc"])
        nf = int(opt["nf"])
        nb = int(opt["nb"])
        ksize = int(opt["ksize"])

        if bool(opt["spatial"]):
            head_k = int(opt["head_k"])
            body_k = int(opt["body_k"])
        else:
            head_k = 1
            body_k = 1

        in_nc = 3 + nc if bool(opt["mix"]) else nc
        deg_kernel = [
            nn.Conv2d(in_nc, nf, head_k, 1, head_k // 2),
            nn.BatchNorm2d(nf),
            nn.ReLU(inplace=True),
            *[PDMResBlock(nf=nf, ksize=body_k) for _ in range(nb)],
            nn.Conv2d(nf, ksize**2, 1, 1, 0),
            nn.Softmax(1),
        ]
        self.deg_kernel = nn.Sequential(*deg_kernel)

        if bool(opt["zero_init"]):
            nn.init.constant_(self.deg_kernel[-2].weight, 0)
            nn.init.constant_(self.deg_kernel[-2].bias, 0)
            with torch.no_grad():
                self.deg_kernel[-2].bias[ksize**2 // 2] = 1

        self.pad = nn.ReflectionPad2d(ksize // 2)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        b, c, h, w = x.shape
        out_h = h // self.scale
        out_w = w // self.scale

        nc = int(self.opt["nc"])
        if nc > 0:
            if bool(self.opt["spatial"]):
                zk = torch.randn(b, nc, out_h, out_w, device=x.device)
            else:
                zk = torch.randn(b, nc, 1, 1, device=x.device)
                if bool(self.opt["mix"]):
                    zk = zk.repeat(1, 1, out_h, out_w)
        else:
            zk = None

        if bool(self.opt["mix"]):
            x_lr = F.interpolate(
                x, size=(out_h, out_w), mode="bicubic", align_corners=False
            )
            inp = torch.cat([x_lr, zk], dim=1) if zk is not None else x_lr
        else:
            if zk is None:
                raise ValueError("PDMKernelModel requires nc>0 when mix is false.")
            inp = zk

        ksize = int(self.opt["ksize"])
        kernel_map = self.deg_kernel(inp)  # [B, k^2, H, W]
        kernel = kernel_map.reshape(b, 1, ksize**2, *kernel_map.shape[2:])

        x_unfold = F.unfold(
            self.pad(x.reshape(b * c, 1, h, w)),
            kernel_size=ksize,
            stride=self.scale,
        ).reshape(b, c, ksize**2, out_h, out_w)
        out = torch.mul(x_unfold, kernel).sum(2).reshape(b, c, out_h, out_w)
        return out, kernel_map


class PDMNoiseModel(nn.Module):
    def __init__(self, opt: dict, scale: int) -> None:
        super().__init__()
        self.scale = scale
        self.opt = opt

        nc = int(opt["nc"])
        nf = int(opt["nf"])
        nb = int(opt["nb"])

        if bool(opt["spatial"]):
            head_k = int(opt["head_k"])
            body_k = int(opt["body_k"])
        else:
            head_k = 1
            body_k = 1

        in_nc = 3 + nc if bool(opt["mix"]) else nc
        deg_noise = [
            nn.Conv2d(in_nc, nf, head_k, 1, head_k // 2),
            nn.BatchNorm2d(nf),
            nn.ReLU(inplace=True),
            *[PDMResBlock(nf=nf, ksize=body_k) for _ in range(nb)],
            nn.Conv2d(nf, int(opt["dim"]), 1, 1, 0),
        ]
        self.deg_noise = nn.Sequential(*deg_noise)

        if bool(opt["zero_init"]):
            nn.init.constant_(self.deg_noise[-1].weight, 0)
            nn.init.constant_(self.deg_noise[-1].bias, 0)
        else:
            nn.init.normal_(self.deg_noise[-1].weight, 0.0, 0.001)
            nn.init.constant_(self.deg_noise[-1].bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        b, _, h, w = x.shape
        if int(self.opt["nc"]) > 0:
            if bool(self.opt["spatial"]):
                zn = torch.randn(b, int(self.opt["nc"]), h, w, device=x.device)
            else:
                zn = torch.randn(b, int(self.opt["nc"]), 1, 1, device=x.device)
                if bool(self.opt["mix"]):
                    zn = zn.repeat(1, 1, h, w)
        else:
            zn = None

        if bool(self.opt["mix"]):
            inp = torch.cat([x, zn], dim=1) if zn is not None else x
        else:
            if zn is None:
                raise ValueError("PDMNoiseModel requires nc>0 when mix is false.")
            inp = zn
        return self.deg_noise(inp)


@ARCH_REGISTRY.register()
class PDMDegModel(nn.Module):
    def __init__(
        self,
        scale: int = 4,
        nc_img: int = 3,
        kernel_opt: dict | None = None,
        noise_opt: dict | None = None,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.nc_img = nc_img
        self.kernel_opt = kernel_opt
        self.noise_opt = noise_opt

        if kernel_opt is not None:
            self.deg_kernel = PDMKernelModel(kernel_opt, scale)
        if noise_opt is not None:
            self.deg_noise = PDMNoiseModel(noise_opt, scale)
        else:
            self.quant = PDMFourierQuantization()

    def forward(self, inp: Tensor) -> tuple[Tensor, Tensor | None, Tensor | None]:
        if self.kernel_opt is not None:
            x, kernel = self.deg_kernel(inp)
        else:
            x = F.interpolate(
                inp, scale_factor=1 / self.scale, mode="bicubic", align_corners=False
            )
            kernel = None

        if self.noise_opt is not None:
            raw_noise = self.deg_noise(x.detach())
            if raw_noise.shape[1] != self.nc_img:
                raise ValueError(
                    "PDM noise output channels must match nc_img. "
                    f"Got noise C={raw_noise.shape[1]} but nc_img={self.nc_img}."
                )

            noise_max = float(self.noise_opt.get("noise_max", 0.05))
            if not (noise_max > 0):
                raise ValueError(f"noise_max must be > 0, got {noise_max}.")

            noise = torch.tanh(raw_noise) * noise_max

            if bool(self.noise_opt.get("zero_mean", False)):
                noise = noise - noise.mean(dim=(2, 3), keepdim=True)
                max_abs = noise.abs().amax(dim=(2, 3), keepdim=True)
                eps = torch.finfo(noise.dtype).eps
                scale = (noise_max / (max_abs + eps)).clamp(max=1.0)
                noise = noise * scale

            x = x + noise
            if bool(self.noise_opt.get("clip_lr", False)):
                x = x.clamp(0.0, 1.0)
        else:
            noise = None
            x = self.quant(x)

        return x, kernel, noise
