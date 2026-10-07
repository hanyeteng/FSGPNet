"""Regression checks for checkpoint loading and released evaluation metrics."""
import unittest
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

from evaluation.mIoU import SamplewiseSigmoidMetric
from evaluation.pd_fa import PD_FA
from evaluation.TPFNFP import SegmentationMetricTPFNFP
from net import Net
from utils.runtime import evaluate, load_checkpoint, resolve_device
from utils.datasets import NUDTSIRSTSetLoader, IRSTD1KSetLoader


class PaddedSample(torch.utils.data.Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, index):
        pred, mask = torch.ones(1, 32, 32), torch.zeros(1, 32, 32)
        pred[:, :8, :8] = 0
        pred[:, 2, 2], mask[:, 2, 2] = 0.8, 1
        return pred, mask, [8, 8], 'sample'


class MetricRegressionTests(unittest.TestCase):
    def test_dataset_binarizes_antialiased_masks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for sub in ('images', 'masks', 'img_idx'):
                (root / sub).mkdir()
            image = np.zeros((32, 32), dtype=np.uint8)
            mask = image.copy()
            mask[0, :4] = [0, 127, 128, 255]
            for sub, value in [('images', image), ('masks', mask)]:
                cv2.imencode('.png', value)[1].tofile(str(root / sub / 'sample.png'))
            for name, loader in [('NUDT-SIRST', NUDTSIRSTSetLoader), ('IRSTD-1K', IRSTD1KSetLoader)]:
                (root / 'img_idx' / f'test_{name}.txt').write_text('sample\n')
                _, label, _, _ = loader(str(root), mode='test')[0]
                self.assertEqual(label[0, 0, :4].tolist(), [0, 0, 1, 1])
                self.assertEqual(set(label.unique().tolist()), {0, 1})

    def test_evaluation_crops_padding_and_restores_training_mode(self):
        model = torch.nn.Identity().train()
        metrics = evaluate(model, PaddedSample(), torch.device('cpu'))
        self.assertTrue(model.training)
        for key in ('mIoU', 'nIoU', 'F1', 'Pd'):
            self.assertAlmostEqual(metrics[key], 1.0)
        self.assertEqual(metrics['Fa'], 0)

    def test_evaluation_restores_mode_after_nonfinite_prediction(self):
        class NonfiniteModel(torch.nn.Module):
            def forward(self, value):
                return value * float('nan')
        model = NonfiniteModel().train()
        with self.assertRaises(FloatingPointError):
            evaluate(model, PaddedSample(), torch.device('cpu'))
        self.assertTrue(model.training)

    def test_niou_obeys_threshold(self):
        pred = torch.tensor([[[[0.4, 0.8]]]])
        label = torch.ones_like(pred)
        for threshold, expected in [(0.5, 0.5), (0.3, 1.0)]:
            metric = SamplewiseSigmoidMetric(1, score_thresh=threshold)
            metric.update(pred, label)
            self.assertAlmostEqual(metric.get(), expected)

    def test_f1_uses_absolute_probabilities(self):
        # Per-image max normalization incorrectly classifies 0.4 as positive.
        metric = SegmentationMetricTPFNFP(1)
        metric.update(np.array([[1, 0]]), np.array([[0.4, 0.1]]))
        self.assertEqual(metric.get_all(), (0, 0, 1))

    def test_f1_obeys_threshold_for_numpy_and_torch(self):
        for as_tensor in (False, True):
            label, pred = np.array([[1, 1]]), np.array([[0.4, 0.8]])
            if as_tensor:
                label, pred = torch.tensor(label), torch.tensor(pred)
            metric = SegmentationMetricTPFNFP(1, threshold=0.3)
            metric.update(label, pred)
            self.assertAlmostEqual(metric.get()[3], 1.0)

    def test_equal_area_false_alarm_is_counted(self):
        pred, label = np.zeros((12, 12)), np.zeros((12, 12))
        pred[2, 2] = pred[9, 9] = label[2, 2] = 1
        metric = PD_FA()
        metric.update(pred, label, (12, 12))
        pd, fa = metric.get()
        self.assertEqual(pd, 1)
        self.assertAlmostEqual(fa, 1 / 144)

    def test_empty_masks_are_finite(self):
        pred = np.zeros((12, 12))
        detection = PD_FA()
        detection.update(pred, pred, pred.shape)
        self.assertEqual(detection.get(), (0, 0))
        segmentation = SegmentationMetricTPFNFP(1)
        segmentation.update(pred, pred)
        self.assertTrue(np.isfinite(segmentation.get()).all())

    def test_detection_obeys_threshold(self):
        pred, label = np.zeros((12, 12)), np.zeros((12, 12))
        pred[2, 2], label[2, 2] = 0.4, 1
        for threshold, expected in [(0.5, 0), (0.3, 1)]:
            metric = PD_FA(threshold=threshold)
            metric.update(pred, label, pred.shape)
            self.assertEqual(metric.get()[0], expected)


class CheckpointRegressionTests(unittest.TestCase):
    def test_supplied_checkpoints_load_strictly_on_cpu(self):
        model = Net("FSGPNet")
        root = Path(__file__).resolve().parents[1]
        for name in ("NUDT-SIRST", "IRSTD-1K"):
            with self.subTest(dataset=name):
                load_checkpoint(model, root / "checkpoints" / name / "FSGPNet.pth.tar")
        for module in model.modules():
            if hasattr(module, "x_grid"):
                self.assertTrue(module.x_grid.is_contiguous())
                self.assertTrue(module.y_grid.is_contiguous())

    def test_cpu_forward_returns_probabilities(self):
        torch.set_num_threads(2)
        model = Net("FSGPNet").eval()
        with torch.inference_mode():
            output = model(torch.zeros(1, 1, 256, 256))
        self.assertEqual(tuple(output.shape), (1, 1, 256, 256))
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(((output >= 0) & (output <= 1)).all())

    def test_unsupported_model_fails_clearly(self):
        with self.assertRaises(ValueError):
            Net("ADGFNet")
        self.assertEqual(resolve_device("cpu"), torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
