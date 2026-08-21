[![arXiv TFM](https://img.shields.io/badge/arXiv-2508.21580-b31b1b.svg)](https://arxiv.org/abs/2508.21580)
[![arXiv CRONOS](https://img.shields.io/badge/arXiv-2512.16577-b31b1b.svg)](https://arxiv.org/abs/2512.16577)
[![ICLR 2026](https://img.shields.io/badge/ICLR-2026-blue)](https://iclr.cc/virtual/2026/poster/10008928)
[![CVPRW 2025](https://img.shields.io/badge/CVPRW%202025-Syndata4CV-1b3d6d)](https://openreview.net/forum?id=sRh6ZMebXJ)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](license.txt)

# Longitudinal4DMed: Models and Tools for Longitudinal  and Spatio Temporal Medical Imaging


This repository is the official implementation of **CRONOS** (ICLR) and the continuations **Temporal Flow Matching (TFM)**, a spatio-temporal and generative 
framework for longitudinal medical imaging. The repository also hosts LAUGEN, a method for generating longitudinal sequences from single images (Syndata4CV @ CVPR 2025).


## Features

- Flow Matching for sequence-to-image forecasting.
- Discrete variant (grid-based, e.g. regular follow-up times).
- Continuous time reconstructions 
- Supports 3D+T or 4D sequences (e.g. MRI volumes, CT or US).
- Simple, dependency-light PyTorch code.
- Supports longitudinal and spatio-temporal medical imaging datasets.


## Status

Actively maintained. Recently added: additional dataloaders, and a nicer eval.py.
Added /laugen for longitudinal augmentations and data generation. 
Coming soon: More baselines. 
## Installation
Clone this repository and install the required packages:
```bash
git clone https://github.com/MIC-DKFZ/Longitudinal4DMed.git
cd Longitudinal4DMed
pip install -e .
```

To launch TensorBoard during or after training:
```bash
tensorboard --logdir checkpoints/logs
```

## Training

Each dataset has a ready-made config in `configs/`:

```bash
# ACDC (cardiac MRI)
python src/train.py --config configs/acdc.yaml

# ISLES 2024 (stroke CTP)
python src/train.py --config configs/isles.yaml

# Lumiere (glioma MRI)
python src/train.py --config configs/lumiere.yaml
```

CLI flags override any value from the config file, e.g. to run a quick debug pass:

```bash
python src/train.py --config configs/acdc.yaml --debug
```

For a quick install check without real data:

```bash
# see src/examples/train_dummy.ipynb
python src/train.py --dummy --device cpu --debug
```

## Contact
For further information, or if you want to reach out to us, visit our  [webpage](https://www.dkfz.de/en/medical-image-computing).

## Citation

If you find this work useful for your research, please consider citing:

```bibtex
@misc{disch2025temporalflowmatchinglearning,
      title={Temporal Flow Matching for Learning Spatio-Temporal Trajectories in 4D Longitudinal Medical Imaging}, 
      author={Nico Albert Disch and Yannick Kirchhoff and Robin Peretzke and Maximilian Rokuss and Saikat Roy and Constantin Ulrich and David Zimmerer and Klaus Maier-Hein},
      year={2025},
      eprint={2508.21580},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2508.21580}, 
}
@misc{disch2025cronoscontinuoustimereconstruction,
      title={CRONOS: Continuous Time Reconstruction for 4D Medical Longitudinal Series}, 
      author={Nico Albert Disch and Saikat Roy and Constantin Ulrich and Yannick Kirchhoff and Maximilian Rokuss and Robin Peretzke and David Zimmerer and Klaus Maier-Hein},
      year={2025},
      eprint={2512.16577},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.16577}, 
}
```
