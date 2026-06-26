from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import torch
from safetensors.torch import load_file
from tqdm import tqdm

from traiNNer.archs import build_network
from traiNNer.utils.color_util import pixelformat2rgb_pt, rgb2pixelformat_pt
from traiNNer.utils.img_util import img2tensor, imwrite, tensor2img
from traiNNer.utils.options import yaml_load
from traiNNer.utils.redux_options import ReduxOptions

#python inference.py -opt options/blind_pdm/train_hat_l_x8.yml -m "traiNNer-redux_UnpairedSR\experiments\blind_pdm_hat_l_x8_noamp_bs2\models\net_g_1000.safetensors" -i "C:\Users\myg\Downloads\lq" -o "C:\Users\myg\Downloads\hq"


VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch SR inference for LQ folder.")
    parser.add_argument(
        "-opt",
        "--opt",
        type=str,
        required=True,
        help="Path to training/test yaml config.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=None,
        help="Path to model weights (.safetensors or .pth). Defaults to path.pretrain_network_g in -opt.",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Input LQ image folder.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output SR folder.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_SR",
        help="Suffix appended to output filename stem.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Inference device, e.g. "cuda", "cuda:0", "cpu". Auto if omitted.',
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Enable strict weight loading. Default: non-strict.",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Disable recursive search in input folder.",
    )
    return parser.parse_args()


def pick_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def remove_common_prefix(
    state_dict: Mapping[str, torch.Tensor], prefixes: tuple[str, ...] = ("module.", "netG.")
) -> dict[str, torch.Tensor]:
    result = dict(state_dict)
    if not result:
        return result
    for prefix in prefixes:
        if all(k.startswith(prefix) for k in result):
            result = {k[len(prefix) :]: v for k, v in result.items()}
    return result


def canonicalize_state_dict(loaded: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    state: Mapping[str, Any] = loaded
    unwrap_keys = (
        "model_state_dict",
        "state_dict",
        "params_ema",
        "params-ema",
        "params",
        "model",
        "net",
    )
    for key in unwrap_keys:
        if key in state and isinstance(state[key], Mapping):
            state = state[key]
            break

    if len(state) == 1:
        single = next(iter(state.values()))
        if isinstance(single, Mapping):
            state = single

    filtered: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if isinstance(value, torch.Tensor):
            filtered[key] = value

    if not filtered:
        raise ValueError("No tensor weights found in checkpoint.")
    return remove_common_prefix(filtered)


def load_generator(
    net_g: torch.nn.Module,
    model_path: Path,
    device: torch.device,
    strict: bool,
) -> None:
    if model_path.suffix.lower() == ".safetensors":
        raw_state: Mapping[str, Any] = load_file(str(model_path), device=str(device))
    elif model_path.suffix.lower() == ".pth":
        raw_state = torch.load(str(model_path), map_location="cpu", weights_only=True)
    else:
        raise ValueError(f"Unsupported weight format: {model_path}")

    state_dict = canonicalize_state_dict(raw_state)
    net_g.load_state_dict(state_dict, strict=strict)


def collect_images(input_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    paths = [p for p in input_dir.glob(pattern) if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS]
    return sorted(paths)


@torch.inference_mode()
def run_one(
    net_g: torch.nn.Module,
    img_path: Path,
    out_path: Path,
    input_pixel_format: str,
    output_pixel_format: str,
    device: torch.device,
) -> None:
    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Failed to read image: {img_path}")

    lq = img2tensor(img_bgr, float32=True, from_bgr=True).unsqueeze(0).to(device)
    net_in = rgb2pixelformat_pt(lq, input_pixel_format)  # type: ignore[arg-type]
    out = net_g(net_in)
    out_rgb = pixelformat2rgb_pt(out, lq, output_pixel_format)  # type: ignore[arg-type]
    out_img = tensor2img(out_rgb, to_bgr=True)
    imwrite(out_img, str(out_path))


def main() -> None:
    args = parse_args()
    opt_path = Path(args.opt)
    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not opt_path.exists():
        raise FileNotFoundError(f"Config not found: {opt_path}")
    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    opt: ReduxOptions
    opt, _ = yaml_load(str(opt_path))
    if opt.network_g is None:
        raise ValueError("network_g is missing in config.")

    device = pick_device(args.device)
    model_path = Path(args.model) if args.model else Path(opt.path.pretrain_network_g or "")
    if not model_path or not model_path.exists():
        raise FileNotFoundError(
            "Model path not found. Pass --model or set path.pretrain_network_g in -opt."
        )

    net_g = build_network({**opt.network_g, "scale": opt.scale}).to(device).eval()
    load_generator(
        net_g=net_g,
        model_path=model_path,
        device=device,
        strict=args.strict,
    )

    image_paths = collect_images(input_dir, recursive=not args.no_recursive)
    if not image_paths:
        raise ValueError(f"No images found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for img_path in tqdm(image_paths, desc="SR", unit="img"):
        rel_parent = img_path.parent.relative_to(input_dir)
        out_dir = output_dir / rel_parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{img_path.stem}{args.suffix}{img_path.suffix}"
        run_one(
            net_g=net_g,
            img_path=img_path,
            out_path=out_path,
            input_pixel_format=opt.input_pixel_format,
            output_pixel_format=opt.output_pixel_format,
            device=device,
        )

    print(f"Done. Saved {len(image_paths)} images to: {output_dir}")


if __name__ == "__main__":
    main()
