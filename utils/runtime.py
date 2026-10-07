"""Shared helpers for training and full-image evaluation."""
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from evaluation.mIoU import mIoU, SamplewiseSigmoidMetric
from evaluation.pd_fa import PD_FA
from evaluation.TPFNFP import SegmentationMetricTPFNFP
from utils.datasets import NUDTSIRSTSetLoader, IRSTD1KSetLoader

DATASETS = ("NUDT-SIRST", "IRSTD-1K")


def resolve_device(value="auto"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if value == "auto" else torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use --device cpu.")
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Supported devices: auto, cpu, cuda, cuda:N.")
    return device


def load_dataset(name, root, mode="test", img_norm_cfg=None):
    loaders = {"NUDT-SIRST": NUDTSIRSTSetLoader, "IRSTD-1K": IRSTD1KSetLoader}
    if name not in loaders:
        raise ValueError(f"Unsupported dataset: {name}")
    base = Path(root) / name
    split = base / "img_idx" / f"{'train' if mode == 'trainval' else 'test'}_{name}.txt"
    if not split.is_file():
        raise FileNotFoundError(f"Missing dataset split: {split}. See README.md for the layout.")
    dataset = loaders[name](base_dir=str(base), mode=mode, img_norm_cfg=img_norm_cfg)
    if not len(dataset):
        raise ValueError(f"Empty dataset split: {split}")
    return dataset


def load_checkpoint(model, path):
    # Only load checkpoints from trusted sources.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    if state and all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    return checkpoint


def evaluate(model, dataset, device, threshold=0.5, workers=0):
    # Different image sizes require one image per batch.
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=workers)
    iou = mIoU()
    niou = SamplewiseSigmoidMetric(nclass=1, score_thresh=threshold)
    detection = PD_FA(threshold=threshold)
    segmentation = SegmentationMetricTPFNFP(nclass=1, threshold=threshold)
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for img, mask, size, _ in loader:
                height, width = int(size[0][0]), int(size[1][0])
                pred = model(img.to(device))
                if getattr(getattr(model, "model", model), "outputs_logits", False):
                    pred = pred.sigmoid()
                pred = pred[:, :, :height, :width].cpu()
                mask = mask[:, :, :height, :width].cpu()
                if not torch.isfinite(pred).all():
                    raise FloatingPointError("Prediction contains NaN or infinity.")
                iou.update(pred > threshold, mask)
                niou.update(pred, mask)
                pred_np, mask_np = pred[0, 0].numpy(), mask[0, 0].numpy()
                detection.update(pred_np, mask_np, (height, width))
                segmentation.update(mask_np, pred_np)
    finally:
        model.train(was_training)
    foreground_recall, miou = iou.get()
    pd, fa = detection.get()
    _, precision, recall, fscore = segmentation.get()
    return dict(mIoU=float(miou), nIoU=float(niou.get()), F1=float(fscore),
                Pd=float(pd), Fa=float(fa), precision=float(precision),
                recall=float(recall), foreground_recall=float(foreground_recall))


def format_results(name, metrics):
    return (f"Results for {name}:\n"
            f"mIoU: {metrics['mIoU']:.6f}, nIoU: {metrics['nIoU']:.6f}, F1: {metrics['F1']:.6f}\n"
            f"Pd: {metrics['Pd']:.6f}, Fa: {metrics['Fa'] * 1e6:.4f}E-6")
