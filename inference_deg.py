"""
推理脚本：使用 net_deg (退化模型) 对 HR 图像进行退化，输出 LR 图像。
参数写在代码顶部，方便测试修改。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import torch
from safetensors.torch import load_file
from tqdm import tqdm

from traiNNer.utils.img_util import img2tensor, imwrite, tensor2img

# ============== 可修改参数（方便测试） ==============
MODEL_PATH = r"C:\Users\myg\Desktop\Unpaired_SR\traiNNer-redux_UnpairedSR\experiments\blind_pdm_hat_l_x8_noiseFix_noamp_bs2\models\resume_models\net_deg_13000.safetensors"
INPUT_DIR = r"C:\Users\myg\Desktop\S2unpairSR\dataset_gen\datasets\s2_arcgis_x8_poc\paired_x8\hr"
OUTPUT_DIR = r"C:\Users\myg\Desktop\S2unpairSR\dataset_gen\datasets\s2_arcgis_x8_poc\paired_x8\hr_deg_output"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SUFFIX = "_deg"
SEED = 42  # 退化模型内部有随机采样，设种子保证可复现
# =================================================

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# network_deg 配置（与 train_hat_l_x8.yml 一致）
NETWORK_DEG_CONFIG = {
    "type": "pdmdegmodel",
    "scale": 8,
    "nc_img": 3,
    "kernel_opt": {
        "spatial": False,
        "mix": False,
        "nc": 3,
        "nf": 64,
        "nb": 8,
        "head_k": 1,
        "body_k": 1,
        "ksize": 21,
        "zero_init": True,
    },
    "noise_opt": {
        "spatial": False,
        "mix": False,
        "nc": 3,
        "nf": 32,
        "nb": 8,
        "head_k": 3,
        "body_k": 3,
        "dim": 1,
        "zero_init": False,
    },
}


def remove_common_prefix(state_dict, prefixes=("module.", "net_deg.")):
    result = dict(state_dict)
    if not result:
        return result
    for prefix in prefixes:
        if all(k.startswith(prefix) for k in result):
            result = {k[len(prefix):]: v for k, v in result.items()}
    return result


def load_deg_model(model_path: Path, device: torch.device):
    from traiNNer.archs import build_network

    raw_state = load_file(str(model_path), device=str(device))
    state = raw_state
    for key in ("model_state_dict", "state_dict", "params_ema", "params-ema", "params", "model", "net"):
        if key in state and isinstance(state[key], dict):
            state = state[key]
            break
    if len(state) == 1:
        single = next(iter(state.values()))
        if isinstance(single, dict):
            state = single
    filtered = {k: v for k, v in state.items() if isinstance(v, torch.Tensor)}
    state_dict = remove_common_prefix(filtered)

    net_deg = build_network(NETWORK_DEG_CONFIG)
    net_deg.load_state_dict(state_dict, strict=False)
    net_deg = net_deg.to(device).eval()
    return net_deg


def collect_images(input_dir: Path):
    paths = [
        p for p in input_dir.glob("**/*")
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    ]
    return sorted(paths)


@torch.inference_mode()
def run_one(net_deg, img_path: Path, out_path: Path, device: torch.device):
    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Failed to read image: {img_path}")

    hr = img2tensor(img_bgr, float32=True, from_bgr=True).unsqueeze(0).to(device)
    lr, _, _ = net_deg(hr)
    out_img = tensor2img(lr, to_bgr=True)
    imwrite(out_img, str(out_path))


def main():
    model_path = Path(MODEL_PATH)
    input_dir = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)

    if not model_path.exists():
        print(f"错误：模型文件不存在: {model_path}")
        sys.exit(1)
    if not input_dir.exists():
        print(f"错误：输入文件夹不存在: {input_dir}")
        sys.exit(1)

    if SEED is not None:
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)

    device = torch.device(DEVICE)
    print(f"加载模型: {model_path}")
    net_deg = load_deg_model(model_path, device)

    image_paths = collect_images(input_dir)
    if not image_paths:
        print(f"错误：在 {input_dir} 中未找到图片")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"输入: {input_dir} ({len(image_paths)} 张)")
    print(f"输出: {output_dir}")

    for img_path in tqdm(image_paths, desc="Deg", unit="img"):
        rel_parent = img_path.parent.relative_to(input_dir)
        out_dir = output_dir / rel_parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{img_path.stem}{SUFFIX}{img_path.suffix}"
        run_one(net_deg, img_path, out_path, device)

    print(f"完成。已保存 {len(image_paths)} 张退化图像到: {output_dir}")


if __name__ == "__main__":
    main()
