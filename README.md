# traiNNer-redux · Unpaired SR

A focused fork of [traiNNer-redux](https://github.com/the-database/traiNNer-redux) dedicated to **unpaired / blind super-resolution**. It keeps traiNNer-redux's large library of network architectures and pairs them with a **Probabilistic Degradation Model (PDM)** so you can train an upscaler **without aligned LR/HR pairs** — you only need a folder of real low-quality images and an unrelated folder of high-quality reference images from the target domain.

All paired-SR, on-the-fly (Real-ESRGAN) degradation, ONNX export, and documentation tooling from upstream has been removed. What remains is the unpaired training pipeline plus every generator/discriminator architecture.

## How unpaired training works

The PDM approach jointly trains four networks:

- **`network_g`** — the SR generator (any architecture below).
- **`network_deg`** (`pdmdegmodel`) — a learned degradation model (blur kernel + noise) that turns HR into realistic synthetic LR.
- **`network_d_lr`** / **`network_d_sr`** (`pdmpatchgandiscriminator`) — PatchGAN discriminators in the LR and SR domains.

The degradation model learns to make synthetic LR match your real LR distribution, while the generator learns to invert it — so no pixel-aligned pairs are required.

Datasets are supplied as unrelated folders:

```yaml
datasets:
  train:
    type: unpairedimagedataset
    dataroot_lq:  [datasets/train/lr]    # real low-quality images
    dataroot_ref: [datasets/train/ref]   # unrelated high-quality target-domain images
```

## Quickstart

```bash
# install (CUDA build of PyTorch recommended)
pip install -e .

# train (PDM unpaired SR)
python train_blind.py --auto_resume -opt options/blind_pdm/train.yml

# Windows convenience launcher
run_train_blind.bat
```

Example configs live in `options/blind_pdm/`:
- `train.yml` — baseline x4 unpaired config (ESRGAN/RRDBNet generator).
- `train_plksr_x4_satellite.yml` — example domain-transfer config (PLKSR generator).

Testing / inference:
```bash
python test_blind.py -opt options/blind_pdm/train.yml      # validation pipeline
python inference.py     ...                                 # run a trained generator on a folder
python inference_deg.py ...                                 # apply a trained degradation model to HR images
```

The `blind.method` option selects the training variant: `pdm_sr` (default) or `pdm_resshift` (ResShift diffusion generator).

## Available architectures

Generators and discriminators are auto-registered from `traiNNer/archs/` (most via [Spandrel](https://github.com/chaiNNer-org/spandrel)). Set `network_g.type` to any of, e.g.:

`esrgan` (RRDBNet), `plksr` / `realplksr`, `span` / `spanplus`, `compact` / `ultracompact`, `swinir_*`, `swin2sr_*`, `hat_*`, `dat` / `dat_2`, `drct`, `rgt`, `srformer`, `atd`, `omnisr`, `man`, `rcan`, `craft`, `elan`, `lmlt`, `mosr` / `moesr2`, `flexnet`, `safmn`, `seemore_t`, `realcugan`, `resshift`, and more (run the registry listing in `traiNNer/archs/__init__.py` to see all entries).

PDM-specific archs: `pdmdegmodel`, `pdmpatchgandiscriminator`. Generic discriminators (`unetdiscriminatorsn`, `vggstylediscriminator`, `patchgandiscriminatorsn`, …) are also available.

## License and acknowledgement

Released under the [Apache License 2.0](LICENSE.txt). See [LICENSE](LICENSE/README.md) for individual licenses and acknowledgements.

- Built on [traiNNer-redux](https://github.com/the-database/traiNNer-redux), itself a fork of [joeyballentine/traiNNer-redux](https://github.com/joeyballentine/traiNNer-redux) and [BasicSR](https://github.com/XPixelGroup/BasicSR).
- Network architectures are imported from [Spandrel](https://github.com/chaiNNer-org/spandrel) and contributors including [umzi2](https://github.com/umzi2), [Kim2091](https://github.com/Kim2091), [stinkybread](https://github.com/stinkybread), and [Artoriuz](https://github.com/Artoriuz).
- The unpaired training pipeline is based on the Probabilistic Degradation Model approach to blind super-resolution.
