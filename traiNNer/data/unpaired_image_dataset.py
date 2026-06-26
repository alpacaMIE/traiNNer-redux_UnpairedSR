import random

import pyvips
from torch import Tensor
from torchvision.transforms.functional import normalize

from traiNNer.data.base_dataset import BaseDataset
from traiNNer.data.transforms import augment, single_random_crop_vips
from traiNNer.utils import FileClient, img2tensor, rgb2ycbcr, scandir
from traiNNer.utils.img_util import img2rgb, vipsimfrompath
from traiNNer.utils.redux_options import DatasetOptions
from traiNNer.utils.registry import DATASET_REGISTRY
from traiNNer.utils.types import DataFeed


def _pad_to_min(img: pyvips.Image, min_size: int) -> pyvips.Image:
    if img.width >= min_size and img.height >= min_size:
        return img
    return img.gravity(
        "centre",
        max(img.width, min_size),
        max(img.height, min_size),
        extend="mirror",
    )


@DATASET_REGISTRY.register()
class UnpairedImageDataset(BaseDataset):
    def __init__(self, opt: DatasetOptions) -> None:
        super().__init__(opt)
        self.file_client = None
        self.io_backend_opt = opt.io_backend
        self.mean = opt.mean
        self.std = opt.std

        assert isinstance(opt.dataroot_lq, list), (
            f"dataroot_lq must be defined for dataset {opt.name}"
        )
        self.lq_folders = opt.dataroot_lq

        ref_roots = (
            opt.dataroot_ref if opt.dataroot_ref is not None else opt.dataroot_gt
        )
        assert isinstance(ref_roots, list), (
            f"dataroot_ref (or dataroot_gt fallback) must be defined for dataset {opt.name}"
        )
        self.ref_folders = ref_roots

        self.lq_paths: list[str] = []
        for folder in self.lq_folders:
            self.lq_paths.extend(
                sorted(scandir(folder, recursive=True, full_path=True))
            )

        self.ref_paths: list[str] = []
        for folder in self.ref_folders:
            self.ref_paths.extend(
                sorted(scandir(folder, recursive=True, full_path=True))
            )

        if len(self.lq_paths) == 0:
            raise ValueError(f"No LQ images found for dataset {opt.name}")
        if len(self.ref_paths) == 0:
            raise ValueError(f"No REF images found for dataset {opt.name}")

        if self.opt.phase != "train" and len(self.lq_paths) != len(self.ref_paths):
            raise ValueError(
                "Validation requires paired lq/ref with equal counts, "
                f"got {len(self.lq_paths)} lq vs {len(self.ref_paths)} ref for {opt.name}"
            )

        self.length = max(len(self.lq_paths), len(self.ref_paths))

    def __getitem__(self, index: int) -> DataFeed:
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop("type"), **self.io_backend_opt
            )

        lq_path = self.lq_paths[index % len(self.lq_paths)]
        if self.opt.phase == "train":
            ref_path = self.ref_paths[random.randint(0, len(self.ref_paths) - 1)]
        else:
            ref_path = self.ref_paths[index % len(self.ref_paths)]

        vips_img_lq = vipsimfrompath(lq_path)
        vips_img_ref = vipsimfrompath(ref_path)

        if self.opt.phase == "train":
            assert self.opt.lq_size is not None, "lq_size is required for train"
            assert self.opt.gt_size is not None, "gt_size is required for train"
            assert self.opt.use_hflip is not None
            assert self.opt.use_rot is not None

            vips_img_lq = _pad_to_min(vips_img_lq, self.opt.lq_size)
            vips_img_ref = _pad_to_min(vips_img_ref, self.opt.gt_size)

            img_lq = single_random_crop_vips(vips_img_lq, self.opt.lq_size)
            img_ref = single_random_crop_vips(vips_img_ref, self.opt.gt_size)
            img_lq = augment(img_lq, self.opt.use_hflip, self.opt.use_rot)
            img_ref = augment(
                img_ref,
                self.opt.use_hflip,
                self.opt.use_rot,
            )
        else:
            img_lq = img2rgb(vips_img_lq.numpy())
            img_ref = img2rgb(vips_img_ref.numpy())

        if self.opt.color == "y":
            img_lq = rgb2ycbcr(img_lq, y_only=True)[..., None]
            img_ref = rgb2ycbcr(img_ref, y_only=True)[..., None]

        lq = img2tensor(img_lq, from_bgr=False, float32=True)
        ref = img2tensor(img_ref, from_bgr=False, float32=True)
        assert isinstance(lq, Tensor)
        assert isinstance(ref, Tensor)

        if self.mean is not None and self.std is not None:
            normalize(lq, self.mean, self.std, inplace=True)
            normalize(ref, self.mean, self.std, inplace=True)

        return {
            "lq": lq,
            "ref": ref,
            "lq_path": lq_path,
            "ref_path": ref_path,
        }

    def __len__(self) -> int:
        return self.length

    @property
    def label(self) -> str:
        return "unpaired images"
