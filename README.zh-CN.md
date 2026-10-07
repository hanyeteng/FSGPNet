# FSGPNet

**Frequency–Spatial Domain Jointly Guided Perceptual Network for Infrared Small Target Detection**

面向红外小目标检测的频率–空间域联合引导感知网络。

[English README](README.md) · [发布检查报告](docs/RELEASE_CHECK.md)

## 方法概览

FSGPNet 以 U-Net 编码器–解码器为基础，组合三个模块：

- **FSEM**：结合风车卷积、Perona–Malik 扩散和基于 DCT 的动态高频感知，增强小目标局部特征。
- **MSGP**：结合非局部注意力、多尺度空洞卷积和通道重标定，建模全局背景。
- **GTAM**：使用 8 个方向、5/7 两种尺度的 Gabor 响应与注意力，筛选跳跃连接中的有效特征。

![网络结构](images/architecture.png)

论文作者：Yeteng Han、Minrui Ye、Bohan Liu、Jie Li、Chaoxian Jia、Wennan Cui、Tao Zhang。题目与作者信息来自提供的校稿后论文，正式出版链接和 DOI 尚未提供。

论文参数量为 4.75M。代码中还保留未调用的 `mtc`、`decoder5` 以兼容已有权重，直接统计全部注册参数为 7.25M；详细统计见发布检查报告。

## 安装

```bash
git clone https://github.com/hanyeteng/FSGPNet.git
cd FSGPNet
```

本地已验证环境：Python 3.10.15、PyTorch 2.0.1、torchvision 0.15.2、CUDA 11.7，环境名 `irsrgan`。

```bash
conda activate irsrgan
python -m pip check
```

新建环境：

```bash
conda create -n fsgpnet python=3.10 -y
conda activate fsgpnet
python -m pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu117
python -m pip install -r requirements.txt
```

仅使用 CPU 时，PyTorch 安装索引改为 `https://download.pytorch.org/whl/cpu`。所有命令从包含 `train.py` 的项目根目录运行。

## 数据与权重

项目包含两份原始预训练权重及四份实验划分文件。数据图片和掩膜不随项目发布。

```text
checkpoints/
  NUDT-SIRST/FSGPNet.pth.tar
  IRSTD-1K/FSGPNet.pth.tar
datasets/
  NUDT-SIRST/
    images/*.png
    masks/*.png
    img_idx/train_NUDT-SIRST.txt
    img_idx/test_NUDT-SIRST.txt
  IRSTD-1K/
    images/*.png
    masks/*.png
    img_idx/train_IRSTD-1K.txt
    img_idx/test_IRSTD-1K.txt
```

每行 ID 不带 `.png`，图像与掩膜同名，掩膜背景/目标取值为 0/255。NUDT-SIRST 划分为 663/664，训练 crop 为 256×256；IRSTD-1K 为 800/201，crop 为 512×512。IRSTD-1K 本地数据共 1001 张，与论文描述的 1000 张有差异，本项目保留提供的实验划分。详细准备规则见 [datasets/README.md](datasets/README.md)。

IRSTD-1K 有 72 张训练掩膜、13 张测试掩膜包含中间灰度边界像素。读取时统一用原始灰度 `>127.5` 二值化，训练和各评估指标使用相同标签。

无需复制本机数据，可直接指定：

```powershell
python test_all.py --dataset_dir "D:\document\ADGFNet\datasets"
```

## 训练

默认设置：Adam、初始学习率 0.001、余弦退火至 0.00001、400 epochs、batch size 8、SoftIoU、阈值 0.5。

```bash
python train.py --dataset_names NUDT-SIRST --save ./log
python train.py --dataset_names IRSTD-1K --save ./log
```

本机外部数据示例：

```powershell
python train.py --dataset_names NUDT-SIRST --dataset_dir "D:\document\ADGFNet\datasets" --save ./log
```

保存目录为 `log/数据集/FSGPNet/`，包括 `last.pth.tar`、`best.pth.tar`、定期 epoch 权重，以及 `config.json`、`metrics.json`、`train.txt`。

```bash
python train.py --dataset_names NUDT-SIRST --resume ./log/NUDT-SIRST/FSGPNet/last.pth.tar
```

续训需使用相同训练计划；如需新计划，使用 `--pretrained` 初始化。原始权重不含完整优化器状态时不能精确续训。随机状态恢复不保证 GPU 逐比特一致。

默认每 10 epochs 和最后一次 epoch 在提供的 **test split** 上评估，不是独立验证集。独立模型选择需要另行准备验证协议。`--amp` 在当前实现中会明确回退 FP32，原因是半精度 DCT/Gabor 训练出现非有限损失。

## 测试

```bash
# 自动匹配两份权重和两个数据集
python test_all.py

# 单数据集
python test.py --dataset_names NUDT-SIRST --weight_path ./checkpoints/NUDT-SIRST/FSGPNet.pth.tar
python test.py --dataset_names IRSTD-1K --weight_path ./checkpoints/IRSTD-1K/FSGPNet.pth.tar

# CPU 执行
python test_all.py --device cpu

# 回归检查
python -m unittest discover -s tests -v
```

完整图像测试只接受 batch size 1；所有指标使用统一绝对概率阈值。终端中的 IoU/nIoU/F1/Pd 为 0–1 小数，Fa 以 10⁻⁶ 为单位，汇总日志写入 `log/`。只加载可信来源的权重。

## 定量结果

**论文结果**（原样摘录，未用本次实测替换）：

| 数据集 | IoU (%) | nIoU (%) | F1 (%) | Pd (%) | Fa (10⁻⁶) |
| --- | ---: | ---: | ---: | ---: | ---: |
| NUDT-SIRST | 94.93 | 94.96 | 97.39 | 98.84 | 2.96 |
| IRSTD-1K | 68.14 | 69.08 | 81.07 | 91.58 | 9.68 |

**当前权重实测**（2026-10-08，RTX 3090 / PyTorch 2.0.1 / FP32 / 阈值 0.5）：

| 数据集 | IoU (%) | nIoU (%) | F1 (%) | Pd (%) | Fa (10⁻⁶) |
| --- | ---: | ---: | ---: | ---: | ---: |
| NUDT-SIRST | 95.96 | 95.95 | 97.94 | 99.26 | 1.4248 |
| IRSTD-1K | 68.44 | 68.74 | 81.26 | 90.91 | 6.0921 |

两套结果不完全一致。本次修复了 F1 按单图最大值归一化及 Fa 按面积误排除假警的问题，并将含中间灰度的掩膜统一二值化；这会影响 IRSTD-1K 部分样本的指标。模型预测保持一致，与论文的差异仍需核对原始实验权重与划分。完整证据与限制见 [发布检查报告](docs/RELEASE_CHECK.md)。本次没有重新训练 400 epochs。

## 定性结果

![论文定性比较](images/qualitative.png)

图像来自提供的论文，不是本次重新生成的比较实验。(a) 输入，(b) ACM，(c) ISTDU-Net，(d) DNANet，(e) UIUNet，(f) AGPCNet，(g) FSGPNet，(h) 真值。

## 引用、来源与许可

临时 BibTeX 见 [英文 README](README.md#citation)，正式出版信息核验后应替换为出版社版本。源码保留 SCTransNet 等原始作者标注；发布前应核对第三方实现的来源和许可。

README 结构参考 [ADGFNet](https://github.com/kuaileBenbi/ADGFNet/blob/main/README.md)。项目尚未指定整体许可证，未擅自声明 MIT 或其他授权。
