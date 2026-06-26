"""Smoke tests for the PDM code-review fixes (docs/PDM_Code_Review_Fixes.md)."""

import sys

import torch

sys.path.insert(0, ".")

from traiNNer.archs.pdm_deg_arch import PDMDegModel, PDMKernelModel, PDMNoiseModel
from traiNNer.archs.pdm_discriminator_arch import PDMPatchGANDiscriminator

torch.manual_seed(0)
failures = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))
    if not cond:
        failures.append(name)


# 1. Kernel model shapes: all (spatial, mix) combos x scales
for spatial in (False, True):
    for mix in (False, True):
        for scale in (1, 2, 4):
            opt = {
                "spatial": spatial, "mix": mix, "nc": 3, "nf": 16, "nb": 1,
                "head_k": 3, "body_k": 3, "ksize": 21, "zero_init": True,
            }
            m = PDMKernelModel(opt, scale)
            x = torch.rand(2, 3, 32, 32)
            try:
                out, kmap = m(x)
                ok = out.shape == (2, 3, 32 // scale, 32 // scale)
                # softmax kernel sums to 1 per spatial position
                ksum = kmap.sum(dim=1)
                ok = ok and torch.allclose(ksum, torch.ones_like(ksum), atol=1e-5)
                check(f"kernel shape spatial={spatial} mix={mix} scale={scale}",
                      ok, f"out={tuple(out.shape)}")
            except Exception as e:  # noqa: BLE001
                check(f"kernel shape spatial={spatial} mix={mix} scale={scale}",
                      False, repr(e))

# 2. Noise init: pre-tanh output should be tiny when zero_init=false
nopt = {
    "spatial": False, "mix": False, "nc": 3, "nf": 32, "nb": 2,
    "head_k": 3, "body_k": 3, "dim": 3, "zero_init": False,
}
nm = PDMNoiseModel(nopt, 2).eval()
with torch.no_grad():
    raw = nm(torch.rand(4, 3, 32, 32))
check("noise init not saturated", raw.abs().max().item() < 0.5,
      f"max|raw|={raw.abs().max().item():.4f}")

# 3. Discriminator outputs 1 channel
d = PDMPatchGANDiscriminator(in_c=3, nf=64, nb=3)
dout = d(torch.rand(2, 3, 64, 64))
check("discriminator 1-channel output", dout.shape[1] == 1, f"shape={tuple(dout.shape)}")

# 4. lr_quant gradient is non-zero after detach fix
sys.path.insert(0, ".")
from traiNNer.models.pdm_sr_blind_model import Quantization  # noqa: E402

quant = Quantization()
x = (torch.rand(1, 3, 8, 8)).requires_grad_(True)
loss = torch.nn.functional.l1_loss(x, quant(x).detach())
loss.backward()
assert x.grad is not None
check("lr_quant grad non-zero", x.grad.abs().sum().item() > 0,
      f"|grad|={x.grad.abs().sum().item():.6f}")

# old buggy form for contrast (should be ~zero)
x2 = (torch.rand(1, 3, 8, 8)).requires_grad_(True)
loss2 = torch.nn.functional.l1_loss(x2, quant(x2))
loss2.backward()
assert x2.grad is not None
check("old form grad is zero (sanity)", x2.grad.abs().sum().item() == 0)

# 5. crop_test starts() logic edge cases (mirrors model code)
def starts(dim: int, crop_size: int) -> list[int]:
    if dim <= crop_size:
        return [0]
    tile_starts = list(range(0, max(dim - crop_size, 0) + 1, crop_size))
    if tile_starts[-1] != dim - crop_size:
        tile_starts.append(dim - crop_size)
    return tile_starts


def coverage_ok(dim: int, crop: int) -> bool:
    cov = [0] * dim
    for s in starts(dim, crop):
        for i in range(s, min(s + crop, dim)):
            cov[i] += 1
    return min(cov) >= 1


for dim, crop in [(64, 64), (128, 64), (100, 64), (65, 64), (63, 64), (200, 64)]:
    check(f"crop_test coverage dim={dim} crop={crop}", coverage_ok(dim, crop))

# 6. Full PDMDegModel forward with the actual yml config values
deg = PDMDegModel(
    scale=2, nc_img=3,
    kernel_opt={"spatial": False, "mix": False, "nc": 3, "nf": 64, "nb": 8,
                "head_k": 1, "body_k": 1, "ksize": 21, "zero_init": True},
    noise_opt={"spatial": False, "mix": False, "nc": 3, "nf": 32, "nb": 8,
               "head_k": 3, "body_k": 3, "dim": 3, "zero_init": False},
)
lr_out, kernel, noise = deg(torch.rand(2, 3, 128, 128))
check("PDMDegModel forward (yml config)",
      lr_out.shape == (2, 3, 64, 64) and noise is not None and kernel is not None,
      f"lr={tuple(lr_out.shape)}")
assert noise is not None
check("noise within noise_max", noise.abs().max().item() <= 0.05 + 1e-6,
      f"max|noise|={noise.abs().max().item():.4f}")

# zero_init: softmax(one-hot center bias=1) over k^2=441 entries gives a
# near-uniform kernel with the center as argmax (matches original PDM init).
deg_k = PDMDegModel(
    scale=2, nc_img=3,
    kernel_opt={"spatial": False, "mix": False, "nc": 3, "nf": 64, "nb": 8,
                "head_k": 1, "body_k": 1, "ksize": 21, "zero_init": True},
    noise_opt=None,
)
deg_k.eval()  # BatchNorm rejects batch=1 with 1x1 spatial in train mode
inp = torch.rand(1, 3, 64, 64)
with torch.no_grad():
    out_k, kmap, _ = deg_k(inp)
assert kmap is not None
center_idx = int(kmap.mean(dim=(0, 2, 3)).argmax().item())
check("zero_init kernel center is argmax", center_idx == 21 * 21 // 2,
      f"argmax={center_idx}, expected {21 * 21 // 2}")

print()
if failures:
    print(f"{len(failures)} FAILURE(S): {failures}")
    sys.exit(1)
print("All smoke tests passed.")
