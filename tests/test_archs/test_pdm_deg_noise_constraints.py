import pytest
import torch

from traiNNer.archs.pdm_deg_arch import PDMDegModel


def test_pdmdeg_noise_constraints() -> None:
    torch.manual_seed(123)

    kernel_opt = {
        "spatial": False,
        "mix": False,
        "nc": 3,
        "nf": 64,
        "nb": 2,
        "head_k": 1,
        "body_k": 1,
        "ksize": 21,
        "zero_init": True,
    }
    noise_opt = {
        "spatial": True,
        "mix": True,
        "nc": 3,
        "nf": 32,
        "nb": 2,
        "head_k": 3,
        "body_k": 3,
        "dim": 3,
        "zero_init": False,
        "noise_max": 0.05,
        "zero_mean": True,
        "clip_lr": True,
    }

    net = PDMDegModel(scale=8, nc_img=3, kernel_opt=kernel_opt, noise_opt=noise_opt).eval()

    inp = torch.rand((2, 3, 64, 64), dtype=torch.float32)
    out, _, noise = net(inp)
    assert noise is not None
    noise_detached = noise.detach()
    out_detached = out.detach()
    assert noise.shape == (2, 3, 8, 8)

    assert float(noise_detached.min()) >= -0.0501
    assert float(noise_detached.max()) <= 0.0501
    assert float(noise_detached.mean(dim=(2, 3)).abs().max()) < 1e-4

    assert float(out_detached.min()) >= 0.0
    assert float(out_detached.max()) <= 1.0


def test_pdmdeg_noise_dim_mismatch_raises() -> None:
    noise_opt = {
        "spatial": True,
        "mix": True,
        "nc": 3,
        "nf": 32,
        "nb": 1,
        "head_k": 3,
        "body_k": 3,
        "dim": 1,
        "zero_init": False,
        "noise_max": 0.05,
        "zero_mean": True,
        "clip_lr": True,
    }
    net = PDMDegModel(scale=8, nc_img=3, kernel_opt=None, noise_opt=noise_opt).eval()
    inp = torch.rand((2, 3, 64, 64), dtype=torch.float32)

    with pytest.raises(ValueError, match="noise output channels"):
        _ = net(inp)
