import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from traiNNer.utils.registry import ARCH_REGISTRY


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * -emb)
        emb = t.float().unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int, time_dim: int) -> None:
        super().__init__()
        self.sinusoidal = SinusoidalPosEmb(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, time_dim),
            nn.SiLU(inplace=True),
            nn.Linear(time_dim, time_dim),
        )

    def forward(self, t: Tensor) -> Tensor:
        return self.mlp(self.sinusoidal(t))


class ResBlock(nn.Module):
    def __init__(
        self, in_ch: int, out_ch: int, time_dim: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x))
        qkv = qkv.view(b, 3, self.num_heads, self.head_dim, h * w)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        attn = F.scaled_dot_product_attention(
            q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2)
        )
        out = attn.transpose(-1, -2).reshape(b, c, h, w)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class EncoderLevel(nn.Module):
    def __init__(
        self,
        blocks: nn.ModuleList,
        downsample: nn.Module | None,
    ) -> None:
        super().__init__()
        self.blocks = blocks
        self.downsample = downsample

    def forward(
        self, h: Tensor, t_emb: Tensor, skips: list[Tensor]
    ) -> Tensor:
        for block in self.blocks:
            h = block(h, t_emb)
            skips.append(h)
        if self.downsample is not None:
            h = self.downsample(h)
        return h


class DecoderLevel(nn.Module):
    def __init__(
        self,
        blocks: nn.ModuleList,
        upsample: nn.Module | None,
    ) -> None:
        super().__init__()
        self.blocks = blocks
        self.upsample = upsample

    def forward(
        self, h: Tensor, t_emb: Tensor, skips: list[Tensor]
    ) -> Tensor:
        for block in self.blocks:
            skip = skips.pop()
            # align spatial dims after upsample (odd-size inputs cause ±1px mismatch)
            if h.shape[-2:] != skip.shape[-2:]:
                h = h[:, :, : skip.shape[2], : skip.shape[3]]
            h = torch.cat([h, skip], dim=1)
            h = block(h, t_emb)
        if self.upsample is not None:
            h = self.upsample(h)
        return h


@ARCH_REGISTRY.register()
class ResShift(nn.Module):
    def __init__(
        self,
        scale: int = 4,
        in_ch: int = 3,
        out_ch: int = 3,
        base_ch: int = 96,
        ch_mult: list[int] | tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        num_heads: int = 4,
        dropout: float = 0.0,
        num_timesteps: int = 15,
        sigma_max: float = 0.35,
        schedule: str = "sine",
    ) -> None:
        super().__init__()
        self.scale = scale
        self.out_ch = out_ch
        self.num_timesteps = num_timesteps
        self.sigma_max = sigma_max
        num_levels = len(ch_mult)
        time_dim = base_ch * 4

        # --- diffusion schedule ---
        self._build_schedule(schedule, num_timesteps, sigma_max)

        # --- time embedding ---
        self.time_embed = TimeEmbedding(base_ch, time_dim)

        # --- input conv: concat(x_t, x_lr_up) ---
        self.conv_in = nn.Conv2d(in_ch * 2, base_ch, 3, padding=1)

        # --- encoder ---
        self.encoder_levels = nn.ModuleList()
        skip_chs: list[int] = []
        cur_ch = base_ch

        for level in range(num_levels):
            out_ch_level = base_ch * ch_mult[level]
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(cur_ch, out_ch_level, time_dim, dropout))
                cur_ch = out_ch_level
                skip_chs.append(cur_ch)
            ds = Downsample(cur_ch) if level < num_levels - 1 else None
            self.encoder_levels.append(EncoderLevel(blocks, ds))

        # --- middle ---
        self.mid_block1 = ResBlock(cur_ch, cur_ch, time_dim, dropout)
        self.mid_attn = AttentionBlock(cur_ch, num_heads)
        self.mid_block2 = ResBlock(cur_ch, cur_ch, time_dim, dropout)

        # --- decoder (mirror encoder) ---
        self.decoder_levels = nn.ModuleList()

        for level in reversed(range(num_levels)):
            out_ch_level = base_ch * ch_mult[level]
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                skip_ch = skip_chs.pop()
                blocks.append(
                    ResBlock(cur_ch + skip_ch, out_ch_level, time_dim, dropout)
                )
                cur_ch = out_ch_level
            us = Upsample(cur_ch) if level > 0 else None
            self.decoder_levels.append(DecoderLevel(blocks, us))

        assert len(skip_chs) == 0, f"Skip channels not fully consumed: {len(skip_chs)} left"

        # --- output ---
        self.norm_out = nn.GroupNorm(32, cur_ch)
        self.conv_out = nn.Conv2d(cur_ch, out_ch, 3, padding=1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def _build_schedule(
        self, schedule: str, num_timesteps: int, sigma_max: float
    ) -> None:
        T = num_timesteps
        steps = torch.arange(T + 1, dtype=torch.float32) / T
        if schedule == "sine":
            mix = torch.sin(steps * math.pi / 2) ** 2
            sigma = sigma_max * torch.sin(steps * math.pi / 2)
        elif schedule == "linear":
            mix = steps
            sigma = sigma_max * steps
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
        self.register_buffer("mix", mix)
        self.register_buffer("sigma", sigma)

    def _unet_forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        h = self.conv_in(x)

        skips: list[Tensor] = []
        for enc_level in self.encoder_levels:
            h = enc_level(h, t_emb, skips)

        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        for dec_level in self.decoder_levels:
            h = dec_level(h, t_emb, skips)

        assert len(skips) == 0
        return self.conv_out(F.silu(self.norm_out(h)))

    # ---- diffusion interface ----

    def q_sample(
        self, x_hr: Tensor, x_lr_up: Tensor, t: Tensor, noise: Tensor | None = None
    ) -> Tensor:
        if noise is None:
            noise = torch.randn_like(x_hr)
        mix_t = self.mix[t].view(-1, 1, 1, 1)
        sigma_t = self.sigma[t].view(-1, 1, 1, 1)
        return (1 - mix_t) * x_hr + mix_t * x_lr_up + sigma_t * noise

    def predict_x0(self, x_t: Tensor, t: Tensor, x_lr_up: Tensor) -> Tensor:
        t_emb = self.time_embed(t)
        inp = torch.cat([x_t, x_lr_up], dim=1)
        return self._unet_forward(inp, t_emb)

    @torch.no_grad()
    def sample(self, x_lr: Tensor) -> Tensor:
        x_lr_up = F.interpolate(
            x_lr, scale_factor=self.scale, mode="bicubic", align_corners=False
        )
        noise = torch.randn_like(x_lr_up)
        x_t = self.mix[-1] * x_lr_up + self.sigma[-1] * noise

        for step in range(self.num_timesteps, 0, -1):
            t = torch.full(
                (x_lr_up.shape[0],), step, device=x_lr_up.device, dtype=torch.long
            )
            x0_pred = self.predict_x0(x_t, t, x_lr_up)

            if step == 1:
                x_t = x0_pred
            else:
                sigma_t = self.sigma[step]
                eps_pred = (
                    x_t - (1 - self.mix[step]) * x0_pred - self.mix[step] * x_lr_up
                ) / sigma_t.clamp_min(1e-6)
                x_t = (
                    (1 - self.mix[step - 1]) * x0_pred
                    + self.mix[step - 1] * x_lr_up
                    + self.sigma[step - 1] * eps_pred
                )

        return x_t.clamp(0, 1)

    def forward(self, x_lr: Tensor) -> Tensor:
        return self.sample(x_lr)
