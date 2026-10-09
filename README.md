# FSGPNet

### Frequency–Spatial Domain Jointly Guided Perceptual Network for Infrared Small Target Detection

[![Project Page](https://img.shields.io/badge/Project-FSGPNet-4c8bf5?style=flat-square)](#fsgpnet)
![Task](https://img.shields.io/badge/Task-Infrared%20Small%20Target%20Detection-orange?style=flat-square)
![Framework](https://img.shields.io/badge/Framework-PyTorch-red?style=flat-square)
[![DOI](https://zenodo.org/badge/DOI/10.3390/rs18071000.svg)](https://doi.org/10.3390/rs18071000)

**Joint frequency and spatial perception for detecting dim infrared targets in cluttered scenes.**

[Overview](#overview) · [Quick Start](#quick-start) · [Results](#quantitative-results) · [Citation](#citation)

> This repository contains FSGPNet training/evaluation code and two pretrained checkpoints. Dataset images and masks must be obtained separately. The supplied checkpoints have been evaluated on the included split lists; those measurements differ from the manuscript's tables. Both sets of results are documented below.

## Overview

FSGPNet combines local detail enhancement, global background modeling, and directional feature selection within a U-Net-style encoder–decoder. It is designed for low-contrast infrared small targets against complex background clutter.

**Paper:** *Frequency–Spatial Domain Jointly Guided Perceptual Network for Infrared Small Target Detection*

**Authors:** Yeteng Han, Minrui Ye, Bohan Liu, Jie Li, Chaoxian Jia, Wennan Cui, and Tao Zhang.

## Quick Start

### 1. Installation

Clone this repository and open a terminal in the repository root, where `train.py` is located:

```bash
git clone https://github.com/hanyeteng/FSGPNet.git
cd FSGPNet
```

To create a separate environment with the checked package versions:

```bash
conda create -n fsgpnet python=3.10 -y
conda activate fsgpnet
python -m pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu117
python -m pip install -r requirements.txt
```

For CPU execution, replace the PyTorch wheel index with `https://download.pytorch.org/whl/cpu`. Choose a compatible PyTorch build for your hardware. CUDA is recommended for full evaluation and training; all entry points also accept `--device cpu`.

### 2. Datasets and Pretrained Weights

The expected directory structure is:

```text
FSGPNet/
├── checkpoints/
│   ├── NUDT-SIRST/FSGPNet.pth.tar
│   └── IRSTD-1K/FSGPNet.pth.tar
├── datasets/
│   ├── NUDT-SIRST/
│   │   ├── images/*.png
│   │   ├── masks/*.png
│   │   └── img_idx/
│   │       ├── train_NUDT-SIRST.txt
│   │       └── test_NUDT-SIRST.txt
│   └── IRSTD-1K/
│       ├── images/*.png
│       ├── masks/*.png
│       └── img_idx/
│           ├── train_IRSTD-1K.txt
│           └── test_IRSTD-1K.txt
├── model/
├── evaluation/
├── utils/
├── train.py
├── test.py
└── test_all.py
```

### 3. Training

```bash
# Train on NUDT-SIRST
python train.py --dataset_names NUDT-SIRST --save ./log

# Train on IRSTD-1K
python train.py --dataset_names IRSTD-1K --save ./log
```

### 4. Evaluation

**Evaluate both supplied checkpoints with one command:**

```bash
python test_all.py
```

The script pairs each dataset with its corresponding checkpoint and prints mIoU, nIoU, F1, Pd, and Fa. A timestamped summary is saved under `log/`.

```bash
# Evaluate one checkpoint
python test.py --dataset_names NUDT-SIRST --weight_path ./checkpoints/NUDT-SIRST/FSGPNet.pth.tar
python test.py --dataset_names IRSTD-1K --weight_path ./checkpoints/IRSTD-1K/FSGPNet.pth.tar
```

## Quantitative Results

Measured on **8 October 2026**, using the supplied weights, the included 663/664 and 800/201 splits, PyTorch 2.0.1 / CUDA 11.7, an RTX 3090, FP32 inference, batch size 1, and threshold 0.5:

| Dataset | IoU (%) | nIoU (%) | F1 (%) | Pd (%) | Fa (10⁻⁶) |
| --- | ---: | ---: | ---: | ---: | ---: |
| NUDT-SIRST | 95.96 | 95.95 | 97.94 | 99.26 | 1.4248 |
| IRSTD-1K | 68.44 | 68.74 | 81.26 | 90.91 | 6.0921 |


## Checkpoint Integrity

| Checkpoint | Size (bytes) |
| --- | ---: |
| `checkpoints/NUDT-SIRST/FSGPNet.pth.tar` | 29.40M | 
| `checkpoints/IRSTD-1K/FSGPNet.pth.tar` | 29.39M | 

## Acknowledgements

The source includes utilities associated with the BasicIRSTD benchmark and SCTransNet configuration/model definitions. Original author attribution has been preserved. The manuscript also discusses pinwheel convolution, high-frequency perception, and context modeling; please consult its bibliography for the research sources. We sincerely thank the BasicIRSTD authors and the authors of all compared methods for making their work publicly available.

## Citation

Use the following provisional record when referring to the supplied manuscript. Replace it with the publisher's official BibTeX when the publication URL, DOI, volume, and article number have been verified.

```bibtex
@article{han2026frequency,
  title={Frequency--spatial domain jointly guided perceptual network for infrared small target detection},
  author={Han, Yeteng and Ye, Minrui and Liu, Bohan and Li, Jie and Jia, Chaoxian and Cui, Wennan and Zhang, Tao},
  journal={Remote Sensing},
  volume={18},
  number={7},
  pages={1000},
  year={2026},
  publisher={MDPI}
}
```

## License

This project is released under the [MIT License](LICENSE).
