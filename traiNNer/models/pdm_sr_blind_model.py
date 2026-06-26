import os
import random
import shutil
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from typing import Any

import cv2
import torch
from torch import Tensor, nn
from torch.amp.grad_scaler import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm.rich import tqdm

from traiNNer.archs import build_network
from traiNNer.data.base_dataset import BaseDataset
from traiNNer.losses import build_loss
from traiNNer.metrics import calculate_metric
from traiNNer.models.base_model import BaseModel
from traiNNer.utils import get_root_logger, imwrite, tensor2img
from traiNNer.utils.color_util import pixelformat2rgb_pt, rgb2pixelformat_pt
from traiNNer.utils.logger import clickable_file_path
from traiNNer.utils.misc import loss_type_to_label
from traiNNer.utils.redux_options import PDMTrainOptions, ReduxOptions
from traiNNer.utils.types import DataFeed


class Quant(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, input: Tensor) -> Tensor:
        output = torch.clamp(input, 0, 1)
        output = (output * 255.0).round() / 255.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tensor:
        return grad_output


class Quantization(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return Quant.apply(input)


class ShuffleBuffer:
    def __init__(self, buffer_size: int) -> None:
        self.buffer_size = buffer_size
        self.num_imgs = 0
        self.images: list[Tensor] = []

    def choose(self, images: Tensor, prob: float = 0.5) -> Tensor:
        if self.buffer_size == 0:
            return images
        return_images: list[Tensor] = []
        for image in images:
            detached_image = torch.unsqueeze(image.detach(), 0)
            if self.num_imgs < self.buffer_size:
                self.images.append(detached_image)
                return_images.append(detached_image)
                self.num_imgs += 1
            elif random.random() < prob:
                idx = random.randint(0, self.buffer_size - 1)
                stored = self.images[idx].clone()
                self.images[idx] = detached_image
                return_images.append(stored)
            else:
                return_images.append(detached_image)
        return torch.cat(return_images, dim=0)


class PDMSRBlindModel(BaseModel):
    def __init__(self, opt: ReduxOptions) -> None:
        super().__init__(opt)
        self.use_amp = self.opt.use_amp
        self.use_channels_last = self.opt.use_channels_last
        self.memory_format = (
            torch.channels_last
            if self.use_amp and self.use_channels_last
            else torch.preserve_format
        )
        self.amp_dtype = torch.bfloat16 if self.opt.amp_bf16 else torch.float16

        assert opt.network_g is not None, "network_g must be defined"
        assert opt.network_deg is not None, "network_deg must be defined"

        self.net_g = build_network({**opt.network_g, "scale": opt.scale})
        self.net_deg = build_network({**opt.network_deg, "scale": opt.scale})

        self.net_d_lr: nn.Module | None = None
        self.net_d_sr: nn.Module | None = None
        if opt.network_d_lr is not None:
            self.net_d_lr = build_network(opt.network_d_lr)
        if opt.network_d_sr is not None:
            self.net_d_sr = build_network(opt.network_d_sr)

        if self.opt.path.pretrain_network_g is not None:
            self.load_network(
                self.net_g,
                self.opt.path.pretrain_network_g,
                self.opt.path.strict_load_g,
                self.opt.path.param_key_g,
            )
        if self.opt.path.pretrain_network_deg is not None:
            self.load_network(
                self.net_deg,
                self.opt.path.pretrain_network_deg,
                self.opt.path.strict_load_deg,
                self.opt.path.param_key_deg,
            )
        if (
            self.net_d_lr is not None
            and self.opt.path.pretrain_network_d_lr is not None
        ):
            self.load_network(
                self.net_d_lr,
                self.opt.path.pretrain_network_d_lr,
                self.opt.path.strict_load_d_lr,
                self.opt.path.param_key_d_lr,
            )
        if (
            self.net_d_sr is not None
            and self.opt.path.pretrain_network_d_sr is not None
        ):
            self.load_network(
                self.net_d_sr,
                self.opt.path.pretrain_network_d_sr,
                self.opt.path.strict_load_d_sr,
                self.opt.path.param_key_d_sr,
            )

        self.net_g = self.model_to_device(self.net_g, compile=self.opt.use_compile)
        self.net_deg = self.model_to_device(self.net_deg)
        if self.net_d_lr is not None:
            self.net_d_lr = self.model_to_device(self.net_d_lr)
        if self.net_d_sr is not None:
            self.net_d_sr = self.model_to_device(self.net_d_sr)

        self.real_lr: Tensor | None = None
        self.syn_hr: Tensor | None = None
        self.fake_real_lr: Tensor | None = None
        self.fake_real_lr_quant: Tensor | None = None
        self.predicted_kernel: Tensor | None = None
        self.predicted_noise: Tensor | None = None
        self.syn_sr: Tensor | None = None
        self.output: Tensor | None = None

        self.losses: dict[str, nn.Module] = {}
        self.quant = Quantization()
        self.optimizer_deg: Optimizer | None = None
        self.optimizer_g: Optimizer | None = None
        self.optimizer_d_lr: Optimizer | None = None
        self.optimizer_d_sr: Optimizer | None = None

        self.pdm_train_opt = PDMTrainOptions()
        self.use_gray_dis = False
        self.d_ratio = 1
        self.optim_deg = True
        self.optim_sr = True
        self.fake_lr_buffer = ShuffleBuffer(0)
        self.fake_hr_buffer = ShuffleBuffer(0)

        if self.is_train and self.opt.train is not None:
            self.init_training_settings()

    def init_training_settings(self) -> None:
        assert self.opt.train is not None
        train_opt = self.opt.train

        self.net_g.train()
        if self.net_d_lr is not None:
            self.net_d_lr.train()
        if self.net_d_sr is not None:
            self.net_d_sr.train()

        self.pdm_train_opt = (
            train_opt.pdm if train_opt.pdm is not None else PDMTrainOptions()
        )
        self.optim_deg = self.pdm_train_opt.optim_deg
        self.optim_sr = self.pdm_train_opt.optim_sr
        if self.optim_deg:
            self.net_deg.train()
        else:
            self.net_deg.eval()
        self.d_ratio = max(1, int(self.pdm_train_opt.d_ratio))
        self.use_gray_dis = self.pdm_train_opt.gray_dis
        self.grad_clip = True
        self.grad_clip_max_norm = self.pdm_train_opt.max_grad_norm
        self.accum_iters = self.opt.datasets["train"].accum_iter
        self.fake_lr_buffer = ShuffleBuffer(self.pdm_train_opt.buffer_size)
        self.fake_hr_buffer = ShuffleBuffer(self.pdm_train_opt.buffer_size)

        enable_gradscaler = self.use_amp and not self.opt.amp_bf16
        self.scaler_ae = GradScaler(enabled=enable_gradscaler, device="cuda")
        self.scaler_g = GradScaler(enabled=enable_gradscaler, device="cuda")
        self.scaler_d = GradScaler(enabled=enable_gradscaler, device="cuda")

        self._setup_losses()
        self._setup_optimizers()
        self.setup_schedulers()

    def _setup_losses(self) -> None:
        assert self.opt.train is not None
        train_opt = self.opt.train
        if train_opt.losses is None:
            raise ValueError("PDM blind training requires train.losses in YAML.")

        for loss_cfg in train_opt.losses:
            cfg = deepcopy(loss_cfg)
            if "type" not in cfg:
                raise ValueError("Each loss config must include type.")
            loss_name = cfg.pop("name", loss_type_to_label(cfg["type"]))
            if float(cfg.get("loss_weight", 1.0)) == 0:
                continue
            loss = build_loss(cfg).to(
                self.device,
                memory_format=self.memory_format,
                non_blocking=True,
            )
            self.losses[loss_name] = loss

        if not self.losses:
            raise ValueError("At least one non-zero weighted loss must be defined.")

    def _append_optimizer(self, optimizer: Optimizer, opt_cfg: dict[str, Any]) -> None:
        self.optimizers.append(optimizer)
        self.optimizers_skipped.append(False)
        self.optimizers_schedule_free.append("SCHEDULEFREE" in opt_cfg["type"].upper())

    def _setup_optimizers(self) -> None:
        assert self.opt.train is not None
        train_opt = self.opt.train
        logger = get_root_logger()

        if self.optim_deg:
            deg_opt_cfg = (
                train_opt.optim_deg
                if train_opt.optim_deg is not None
                else train_opt.optim_g
            )
            if deg_opt_cfg is None:
                raise ValueError(
                    "optim_deg is enabled but train.optim_deg/optim_g is missing."
                )
            self.optimizer_deg = self.get_optimizer(
                self.net_deg.parameters(), deg_opt_cfg
            )
            self._append_optimizer(self.optimizer_deg, deg_opt_cfg)
        else:
            logger.warning("optim_deg is disabled. net_deg will not be updated.")

        if self.optim_sr:
            if train_opt.optim_g is None:
                raise ValueError("optim_sr is enabled but train.optim_g is missing.")
            self.optimizer_g = self.get_optimizer(
                self.net_g.parameters(), train_opt.optim_g
            )
            self._append_optimizer(self.optimizer_g, train_opt.optim_g)
        else:
            logger.warning("optim_sr is disabled. net_g will not be updated.")

        if self.net_d_lr is not None and "lr_adv" in self.losses:
            d_lr_opt_cfg = (
                train_opt.optim_d_lr
                if train_opt.optim_d_lr is not None
                else train_opt.optim_d
            )
            if d_lr_opt_cfg is None:
                raise ValueError(
                    "lr_adv is enabled but train.optim_d_lr/optim_d is missing."
                )
            self.optimizer_d_lr = self.get_optimizer(
                self.net_d_lr.parameters(), d_lr_opt_cfg
            )
            self._append_optimizer(self.optimizer_d_lr, d_lr_opt_cfg)

        if self.net_d_sr is not None and "sr_adv" in self.losses:
            d_sr_opt_cfg = (
                train_opt.optim_d_sr
                if train_opt.optim_d_sr is not None
                else train_opt.optim_d
            )
            if d_sr_opt_cfg is None:
                raise ValueError(
                    "sr_adv is enabled but train.optim_d_sr/optim_d is missing."
                )
            self.optimizer_d_sr = self.get_optimizer(
                self.net_d_sr.parameters(), d_sr_opt_cfg
            )
            self._append_optimizer(self.optimizer_d_sr, d_sr_opt_cfg)

    def feed_data(self, data: DataFeed) -> None:
        assert "lq" in data
        self.real_lr = data["lq"].to(
            self.device, memory_format=self.memory_format, non_blocking=True
        )

        if "ref" in data:
            self.syn_hr = data["ref"].to(
                self.device, memory_format=self.memory_format, non_blocking=True
            )
        elif "gt" in data:
            self.syn_hr = data["gt"].to(
                self.device, memory_format=self.memory_format, non_blocking=True
            )
        else:
            self.syn_hr = None

    def _to_gray(self, x: Tensor) -> Tensor:
        if x.shape[1] != 3:
            return x
        return (
            0.299 * x[:, 0:1, :, :] + 0.587 * x[:, 1:2, :, :] + 0.114 * x[:, 2:3, :, :]
        )

    def _set_requires_grad(self, net: nn.Module | None, requires_grad: bool) -> None:
        if net is None:
            return
        for p in net.parameters():
            p.requires_grad = requires_grad

    def _deg_forward(self) -> None:
        assert self.syn_hr is not None
        self.fake_real_lr, self.predicted_kernel, self.predicted_noise = self.net_deg(
            self.syn_hr
        )
        if "sr_pix_trans" in self.losses:
            self.fake_real_lr_quant = self.quant(self.fake_real_lr)
            net_g_in = rgb2pixelformat_pt(
                self.fake_real_lr_quant, self.opt.input_pixel_format
            )
            with torch.autocast(
                device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp
            ):
                self.syn_sr = self.net_g(net_g_in)
                self.syn_sr = pixelformat2rgb_pt(
                    self.syn_sr, self.syn_hr, self.opt.output_pixel_format
                )

    def _sr_forward(self) -> None:
        assert self.syn_hr is not None
        if (not self.optim_deg) or self.fake_real_lr is None:
            self.fake_real_lr, self.predicted_kernel, self.predicted_noise = (
                self.net_deg(self.syn_hr)
            )

        self.fake_real_lr_quant = self.quant(self.fake_real_lr)
        net_g_in = rgb2pixelformat_pt(
            self.fake_real_lr_quant.detach(), self.opt.input_pixel_format
        )
        out = self.net_g(net_g_in)
        self.syn_sr = pixelformat2rgb_pt(out, self.syn_hr, self.opt.output_pixel_format)
        self.output = self.syn_sr

    def _gan_g_loss(
        self, net_d: nn.Module, criterion: nn.Module, fake: Tensor
    ) -> Tensor:
        d_pred_fake = net_d(fake)
        return criterion(d_pred_fake, True, is_disc=False)

    def _gan_d_loss(
        self, net_d: nn.Module, criterion: nn.Module, real: Tensor, fake: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        d_pred_fake = net_d(fake.detach())
        d_pred_real = net_d(real)
        loss_real = criterion(d_pred_real, True, is_disc=True)
        loss_fake = criterion(d_pred_fake, False, is_disc=True)
        return (
            (loss_real + loss_fake) / 2,
            d_pred_real.detach().mean(),
            d_pred_fake.detach().mean(),
        )

    def _clip_and_step(
        self,
        optimizer: Optimizer,
        scaler: GradScaler,
        parameters: Any,
        skip_idx: int,
        apply_gradient: bool,
    ) -> Tensor | None:
        if not apply_gradient:
            return None
        params = list(parameters)
        scaler.unscale_(optimizer)
        grads = [
            torch.linalg.vector_norm(p.grad, 2) for p in params if p.grad is not None
        ]
        grad_norm = (
            torch.linalg.vector_norm(torch.stack(grads)).detach()
            if grads
            else torch.tensor(0.0, device=self.device)
        )
        if self.grad_clip:
            clip_grad_norm_(params, self.grad_clip_max_norm)
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        scale_after = scaler.get_scale()
        if skip_idx < len(self.optimizers_skipped):
            self.optimizers_skipped[skip_idx] = scale_after < scale_before
        optimizer.zero_grad()
        return grad_norm

    def _optimize_trans_models(
        self,
        current_iter: int,
        apply_gradient: bool,
        loss_dict: dict[str, Tensor | float],
    ) -> None:
        if self.optimizer_deg is None:
            return
        assert self.real_lr is not None and self.syn_hr is not None
        assert self.scaler_ae is not None

        self._set_requires_grad(self.net_deg, True)
        self._set_requires_grad(self.net_g, False)
        self._set_requires_grad(self.net_d_lr, False)

        with torch.autocast(
            device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp
        ):
            self._deg_forward()
            loss_g = torch.tensor(0.0, device=self.device)

            if "lr_adv" in self.losses and self.net_d_lr is not None:
                real = (
                    self._to_gray(self.real_lr) if self.use_gray_dis else self.real_lr
                )
                fake = (
                    self._to_gray(self.fake_real_lr)
                    if self.use_gray_dis
                    else self.fake_real_lr
                )
                g1_adv = self._gan_g_loss(self.net_d_lr, self.losses["lr_adv"], fake)
                loss_dict["g1_adv"] = g1_adv
                loss_g = loss_g + self.losses["lr_adv"].loss_weight * g1_adv

            if "sr_pix_trans" in self.losses:
                assert self.syn_sr is not None
                sr_pix = self.losses["sr_pix_trans"](self.syn_sr, self.syn_hr)
                loss_dict["sr_pix_trans"] = sr_pix
                loss_g = loss_g + self.losses["sr_pix_trans"].loss_weight * sr_pix

            if "noise_mean" in self.losses and self.predicted_noise is not None:
                noise_mean = self.losses["noise_mean"](
                    self.predicted_noise, torch.zeros_like(self.predicted_noise)
                )
                loss_dict["noise_mean"] = noise_mean
                loss_g = loss_g + self.losses["noise_mean"].loss_weight * noise_mean

            if "lr_gauss" in self.losses and self.predicted_kernel is not None:
                lr_gauss = self.losses["lr_gauss"](self.predicted_kernel)
                loss_dict["lr_gauss"] = lr_gauss
                loss_g = loss_g + self.losses["lr_gauss"].loss_weight * lr_gauss

            if "lr_quant" in self.losses:
                lr_quant = self.losses["lr_quant"](
                    self.fake_real_lr, self.quant(self.fake_real_lr).detach()
                )
                loss_dict["lr_quant"] = lr_quant
                loss_g = loss_g + self.losses["lr_quant"].loss_weight * lr_quant

            loss_g = loss_g / self.accum_iters
            loss_dict["l_trans_total"] = loss_g

        self.scaler_ae.scale(loss_g).backward()
        grad_norm_deg = self._clip_and_step(
            optimizer=self.optimizer_deg,
            scaler=self.scaler_ae,
            parameters=self.net_deg.parameters(),
            skip_idx=0,
            apply_gradient=apply_gradient,
        )
        if grad_norm_deg is not None:
            loss_dict["grad_norm_deg_pre_clip"] = grad_norm_deg
            loss_dict["grad_clipped_deg"] = float(
                (grad_norm_deg > self.grad_clip_max_norm).item()
            )

        if (
            apply_gradient
            and current_iter % self.d_ratio == 0
            and "lr_adv" in self.losses
            and self.net_d_lr is not None
            and self.optimizer_d_lr is not None
            and self.scaler_d is not None
        ):
            self._set_requires_grad(self.net_d_lr, True)
            real = self._to_gray(self.real_lr) if self.use_gray_dis else self.real_lr
            fake = (
                self._to_gray(self.fake_real_lr)
                if self.use_gray_dis
                else self.fake_real_lr
            )
            with torch.autocast(
                device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp
            ):
                d_loss, out_real, out_fake = self._gan_d_loss(
                    self.net_d_lr,
                    self.losses["lr_adv"],
                    real,
                    self.fake_lr_buffer.choose(fake),
                )
                loss_dict["d1_adv"] = d_loss
                loss_dict["out_d1_real"] = out_real
                loss_dict["out_d1_fake"] = out_fake
                d_loss = d_loss / self.accum_iters

            self.scaler_d.scale(d_loss).backward()
            grad_norm_d1 = self._clip_and_step(
                optimizer=self.optimizer_d_lr,
                scaler=self.scaler_d,
                parameters=self.net_d_lr.parameters(),
                skip_idx=2 if self.optimizer_g is not None else 1,
                apply_gradient=True,
            )
            if grad_norm_d1 is not None:
                loss_dict["grad_norm_d1_pre_clip"] = grad_norm_d1
                loss_dict["grad_clipped_d1"] = float(
                    (grad_norm_d1 > self.grad_clip_max_norm).item()
                )

    def _optimize_sr_models(
        self,
        current_iter: int,
        apply_gradient: bool,
        loss_dict: dict[str, Tensor | float],
    ) -> None:
        if self.optimizer_g is None:
            return
        assert self.syn_hr is not None
        assert self.scaler_g is not None

        self._set_requires_grad(self.net_g, True)
        self._set_requires_grad(self.net_deg, False)
        self._set_requires_grad(self.net_d_sr, False)

        with torch.autocast(
            device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp
        ):
            self._sr_forward()
            assert self.syn_sr is not None

            loss_g = torch.tensor(0.0, device=self.device)
            if "sr_adv" in self.losses and self.net_d_sr is not None:
                sr_adv = self._gan_g_loss(
                    self.net_d_sr, self.losses["sr_adv"], self.syn_sr
                )
                loss_dict["sr_adv"] = sr_adv
                loss_g = loss_g + self.losses["sr_adv"].loss_weight * sr_adv

            if "sr_percep" in self.losses:
                sr_percep = self.losses["sr_percep"](self.syn_sr, self.syn_hr)
                if isinstance(sr_percep, tuple):
                    percep, style = sr_percep
                    if percep is not None:
                        loss_dict["sr_percep"] = percep
                        loss_g = loss_g + self.losses["sr_percep"].loss_weight * percep
                    if style is not None:
                        loss_dict["sr_style"] = style
                        loss_g = loss_g + self.losses["sr_percep"].loss_weight * style
                elif isinstance(sr_percep, dict):
                    sr_percep_total = torch.tensor(0.0, device=self.device)
                    for layer_name, layer_loss in sr_percep.items():
                        loss_dict[f"sr_percep_{layer_name}"] = layer_loss
                        sr_percep_total = sr_percep_total + layer_loss
                    loss_g = (
                        loss_g + self.losses["sr_percep"].loss_weight * sr_percep_total
                    )
                else:
                    loss_dict["sr_percep"] = sr_percep
                    loss_g = loss_g + self.losses["sr_percep"].loss_weight * sr_percep

            if "sr_pix_sr" in self.losses:
                sr_pix = self.losses["sr_pix_sr"](self.syn_sr, self.syn_hr)
                loss_dict["sr_pix_sr"] = sr_pix
                loss_g = loss_g + self.losses["sr_pix_sr"].loss_weight * sr_pix

            if "color" in self.losses:
                color_loss = self.losses["color"](self.syn_sr, self.syn_hr)
                loss_dict["color"] = color_loss
                loss_g = loss_g + self.losses["color"].loss_weight * color_loss

            loss_g = loss_g / self.accum_iters
            loss_dict["l_sr_total"] = loss_g

        self.scaler_g.scale(loss_g).backward()
        grad_norm_g = self._clip_and_step(
            optimizer=self.optimizer_g,
            scaler=self.scaler_g,
            parameters=self.net_g.parameters(),
            skip_idx=1 if self.optimizer_deg is not None else 0,
            apply_gradient=apply_gradient,
        )
        if grad_norm_g is not None:
            loss_dict["grad_norm_g_pre_clip"] = grad_norm_g
            loss_dict["grad_clipped_g"] = float(
                (grad_norm_g > self.grad_clip_max_norm).item()
            )

        if (
            apply_gradient
            and current_iter % self.d_ratio == 0
            and "sr_adv" in self.losses
            and self.net_d_sr is not None
            and self.optimizer_d_sr is not None
            and self.scaler_d is not None
        ):
            self._set_requires_grad(self.net_d_sr, True)
            with torch.autocast(
                device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp
            ):
                d2_loss, out_real, out_fake = self._gan_d_loss(
                    self.net_d_sr,
                    self.losses["sr_adv"],
                    self.syn_hr,
                    self.fake_hr_buffer.choose(self.syn_sr),
                )
                loss_dict["d2_adv"] = d2_loss
                loss_dict["out_d2_real"] = out_real
                loss_dict["out_d2_fake"] = out_fake
                d2_loss = d2_loss / self.accum_iters

            self.scaler_d.scale(d2_loss).backward()
            idx = len(self.optimizers) - 1
            grad_norm_d2 = self._clip_and_step(
                optimizer=self.optimizer_d_sr,
                scaler=self.scaler_d,
                parameters=self.net_d_sr.parameters(),
                skip_idx=idx,
                apply_gradient=True,
            )
            if grad_norm_d2 is not None:
                loss_dict["grad_norm_d2_pre_clip"] = grad_norm_d2
                loss_dict["grad_clipped_d2"] = float(
                    (grad_norm_d2 > self.grad_clip_max_norm).item()
                )

    def optimize_parameters(
        self, current_iter: int, current_accum_iter: int, apply_gradient: bool
    ) -> None:
        del current_accum_iter
        assert self.real_lr is not None
        n_samples = self.real_lr.shape[0]
        self.loss_samples += n_samples

        loss_dict: dict[str, Tensor | float] = OrderedDict()
        if self.optim_deg:
            self._optimize_trans_models(current_iter, apply_gradient, loss_dict)
        if self.optim_sr and self.syn_hr is not None:
            self._optimize_sr_models(current_iter, apply_gradient, loss_dict)

        for key, value in loss_dict.items():
            val = (
                value
                if isinstance(value, float)
                else value.to(dtype=torch.float32).detach()
            )
            self.log_dict[key] = self.log_dict.get(key, 0) + val * n_samples
        self.log_dict = self.reduce_loss_dict(self.log_dict)

    def infer_tiled(self, net: nn.Module, lq: Tensor) -> Tensor:
        assert self.opt.val is not None
        tile_size = self.opt.val.tile_size
        tile_overlap = self.opt.val.tile_overlap
        scale = self.opt.scale

        b, c, h, w = lq.shape
        assert b == 1, "Only batch size 1 is supported for tiled inference"

        if h <= tile_size and w <= tile_size:
            return net(lq)

        pad_h = (tile_size - (h % tile_size)) % tile_size if h > tile_size else 0
        pad_w = (tile_size - (w % tile_size)) % tile_size if w > tile_size else 0
        lq = torch.nn.functional.pad(lq, (0, pad_w, 0, pad_h), mode="reflect")
        _, _, h_pad, w_pad = lq.shape

        output = torch.zeros((1, c, h_pad * scale, w_pad * scale), device=lq.device)
        weight_map = torch.zeros_like(output)

        hr_tile = tile_size * scale
        wy = torch.linspace(0, 1, hr_tile, device=lq.device)
        wx = torch.linspace(0, 1, hr_tile, device=lq.device)
        wy = (1 - torch.abs(wy - 0.5) * 2).clamp(min=0.01)
        wx = (1 - torch.abs(wx - 0.5) * 2).clamp(min=0.01)
        weight = torch.ger(wy, wx).unsqueeze(0).unsqueeze(0)

        stride = tile_size - tile_overlap
        tiles_y = max(1, (h_pad - tile_overlap + stride - 1) // stride)
        tiles_x = max(1, (w_pad - tile_overlap + stride - 1) // stride)

        for y in range(tiles_y):
            for x in range(tiles_x):
                in_y0 = y * stride
                in_x0 = x * stride
                in_y1 = min(in_y0 + tile_size, h_pad)
                in_x1 = min(in_x0 + tile_size, w_pad)
                lq_patch = lq[:, :, in_y0:in_y1, in_x0:in_x1]
                ph, pw = lq_patch.shape[-2:]
                out_patch = net(lq_patch)[:, :, : ph * scale, : pw * scale]
                w_patch = weight[:, :, : ph * scale, : pw * scale]

                out_y0 = in_y0 * scale
                out_x0 = in_x0 * scale
                out_y1 = out_y0 + ph * scale
                out_x1 = out_x0 + pw * scale
                output[:, :, out_y0:out_y1, out_x0:out_x1] += out_patch * w_patch
                weight_map[:, :, out_y0:out_y1, out_x0:out_x1] += w_patch

        out_final = output / weight_map.clamp(min=1e-6)
        return out_final[:, :, : h * scale, : w * scale]

    def crop_test(self, lr: Tensor, crop_size: int) -> Tensor:
        b, c, h, w = lr.shape
        scale = self.opt.scale

        if h <= crop_size and w <= crop_size:
            return self.net_g(lr)

        def starts(dim: int) -> list[int]:
            if dim <= crop_size:
                return [0]
            tile_starts = list(range(0, max(dim - crop_size, 0) + 1, crop_size))
            if tile_starts[-1] != dim - crop_size:
                tile_starts.append(dim - crop_size)
            return tile_starts

        output = torch.zeros(b, c, h * scale, w * scale, device=self.device)
        count = torch.zeros(1, 1, h * scale, w * scale, device=self.device)

        for hs in starts(h):
            for ws in starts(w):
                lr_patch = lr[:, :, hs : hs + crop_size, ws : ws + crop_size]
                sr_patch = self.net_g(lr_patch)
                ph, pw = lr_patch.shape[-2:]
                ys, xs = hs * scale, ws * scale
                ye, xe = ys + ph * scale, xs + pw * scale
                output[:, :, ys:ye, xs:xe] += sr_patch[:, :, : ph * scale, : pw * scale]
                count[:, :, ys:ye, xs:xe] += 1

        return output / count

    def test(self) -> None:
        assert self.real_lr is not None
        lq = rgb2pixelformat_pt(self.real_lr, self.opt.input_pixel_format)

        g_was_training = self.net_g.training
        self.net_g.eval()
        with torch.inference_mode():
            if self.opt.val is not None and self.opt.val.tile_size > 0:
                out = self.infer_tiled(self.net_g, lq)
            elif (
                self.opt.val is not None
                and self.opt.val.pdm is not None
                and self.opt.val.pdm.crop_test
            ):
                if self.opt.val.pdm.crop_size <= 0:
                    raise ValueError(
                        "val.pdm.crop_size must be > 0 when crop_test is enabled."
                    )
                out = self.crop_test(lq, self.opt.val.pdm.crop_size)
            else:
                out = self.net_g(lq)
            self.output = pixelformat2rgb_pt(
                out, self.syn_hr, self.opt.output_pixel_format
            )
        if g_was_training:
            self.net_g.train()

        if self.syn_hr is not None:
            deg_was_training = self.net_deg.training
            self.net_deg.eval()
            with torch.inference_mode():
                self.fake_real_lr = self.net_deg(self.syn_hr)[0]
            if deg_was_training:
                self.net_deg.train()

    def dist_validation(
        self,
        dataloader: DataLoader,
        current_iter: int,
        tb_logger: SummaryWriter | None,
        save_img: bool,
        multi_val_datasets: bool,
    ) -> None:
        if self.opt.rank == 0:
            self.nondist_validation(
                dataloader, current_iter, tb_logger, save_img, multi_val_datasets
            )

    def nondist_validation(
        self,
        dataloader: DataLoader,
        current_iter: int,
        tb_logger: SummaryWriter | None,
        save_img: bool,
        multi_val_datasets: bool,
    ) -> None:
        self.is_train = False
        assert isinstance(dataloader.dataset, BaseDataset)
        assert self.opt.path.visualization is not None

        dataset_name = dataloader.dataset.opt.name
        run_metrics = self.with_metrics
        if run_metrics:
            assert self.opt.val is not None and self.opt.val.metrics is not None
            if len(self.metric_results) == 0:
                self.metric_results = dict.fromkeys(self.opt.val.metrics.keys(), 0)
            self._initialize_best_metric_results(dataset_name)
            self.metric_results = dict.fromkeys(self.metric_results, 0)
        metric_count = 0

        metric_data: dict[str, Any] = {}
        pbar = tqdm(total=len(dataloader), unit="image") if self.use_pbar else None
        logger = get_root_logger()
        if save_img and len(dataloader) > 0:
            logger.info(
                "Saving %d validation images to %s.",
                len(dataloader),
                clickable_file_path(
                    self.opt.path.visualization, "visualization folder"
                ),
            )

        gt_key = "img2"
        for val_data in dataloader:
            img_name = osp.splitext(osp.basename(val_data["lq_path"][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img(visuals["result"], to_bgr=False)
            metric_data["img"] = sr_img
            has_ref = "ref" in visuals
            if has_ref:
                ref_img = tensor2img(visuals["ref"], to_bgr=False)
                metric_data[gt_key] = ref_img
            else:
                metric_data.pop(gt_key, None)

            self.real_lr = None
            self.output = None

            save_img_dir = None
            if save_img:
                assert self.opt.val is not None
                if self.opt.is_train:
                    if multi_val_datasets:
                        save_img_dir = osp.join(
                            self.opt.path.visualization, f"{dataset_name} - {img_name}"
                        )
                    else:
                        assert dataloader.dataset.opt.dataroot_lq is not None
                        lq_path = val_data["lq_path"][0]
                        normalized_lq_path = osp.normpath(lq_path)
                        matching_root = None
                        for root in dataloader.dataset.opt.dataroot_lq:
                            normalized_root = osp.normpath(root)
                            if normalized_lq_path.startswith(normalized_root + osp.sep):
                                matching_root = root
                                break
                        if matching_root is None:
                            raise ValueError(
                                f"The lq_path {lq_path} does not match any dataroot_lq."
                            )
                        save_img_dir = osp.join(
                            self.opt.path.visualization,
                            osp.relpath(osp.splitext(lq_path)[0], matching_root),
                        )
                    save_img_path = osp.join(
                        save_img_dir, f"{img_name}_{current_iter:06d}.png"
                    )
                elif self.opt.val.suffix:
                    save_img_path = osp.join(
                        self.opt.path.visualization,
                        dataset_name,
                        f"{img_name}_{self.opt.val.suffix}.png",
                    )
                else:
                    save_img_path = osp.join(
                        self.opt.path.visualization, dataset_name, f"{img_name}.png"
                    )
                imwrite(cv2.cvtColor(sr_img, cv2.COLOR_RGB2BGR), save_img_path)
                if (
                    self.opt.is_train
                    and not self.first_val_completed
                    and "lq_path" in val_data
                ):
                    assert save_img_dir is not None
                    lr_img_target_path = osp.join(save_img_dir, f"{img_name}_lr.png")
                    if not os.path.exists(lr_img_target_path):
                        shutil.copy(val_data["lq_path"][0], lr_img_target_path)

            if run_metrics and has_ref:
                assert self.opt.val is not None and self.opt.val.metrics is not None
                for name, opt_ in self.opt.val.metrics.items():
                    self.metric_results[name] += calculate_metric(
                        metric_data, opt_, self.device
                    )
                metric_count += 1

            if pbar is not None:
                pbar.update(1)
                pbar.set_description(f"Test {img_name}")

        if pbar is not None:
            pbar.close()

        if run_metrics and metric_count > 0:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= metric_count
                self._update_best_metric_result(
                    dataset_name, metric, self.metric_results[metric], current_iter
                )
            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

        self.first_val_completed = True
        self.is_train = True

    def _log_validation_metric_values(
        self, current_iter: int, dataset_name: str, tb_logger: SummaryWriter | None
    ) -> None:
        log_str = f"Validation {dataset_name}\n"
        for metric, value in self.metric_results.items():
            log_str += f"\t # {metric:<5}: {value:7.4f}"
            if len(self.best_metric_results) > 0:
                log_str += (
                    f"\tBest: {self.best_metric_results[dataset_name][metric]['val']:7.4f} @ "
                    f"{self.best_metric_results[dataset_name][metric]['iter']:9,} iter"
                )
            log_str += "\n"
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(
                    f"metrics/{dataset_name}/{metric}", value, current_iter
                )

    def get_current_visuals(self) -> dict[str, Tensor]:
        assert self.output is not None
        assert self.real_lr is not None
        out_dict: dict[str, Tensor] = OrderedDict()
        out_dict["lq"] = self.real_lr.detach().cpu()
        out_dict["result"] = self.output.detach().cpu()
        if self.syn_hr is not None:
            out_dict["ref"] = self.syn_hr.detach().cpu()
        if self.fake_real_lr is not None:
            out_dict["fake_lr"] = self.fake_real_lr.detach().cpu()
        return out_dict

    def save(self, epoch: int, current_iter: int) -> None:
        assert self.opt.path.models is not None
        assert self.opt.path.resume_models is not None

        self.save_network(
            self.net_g, "net_g", self.opt.path.models, current_iter, "params"
        )
        self.save_network(
            self.net_deg, "net_deg", self.opt.path.resume_models, current_iter, "params"
        )
        if self.net_d_lr is not None:
            self.save_network(
                self.net_d_lr,
                "net_d_lr",
                self.opt.path.resume_models,
                current_iter,
                "params",
            )
        if self.net_d_sr is not None:
            self.save_network(
                self.net_d_sr,
                "net_d_sr",
                self.opt.path.resume_models,
                current_iter,
                "params",
            )
        self.save_training_state(epoch, current_iter)
