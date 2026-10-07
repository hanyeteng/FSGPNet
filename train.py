"""Train FSGPNet with SoftIoU and the manuscript's default schedule."""
import argparse
import json
import random
import warnings
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from net import Net
from utils.runtime import DATASETS, evaluate, format_results, load_checkpoint, load_dataset, resolve_device
from utils.utils import get_optimizer, seed_pytorch


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_names', nargs='+', choices=['FSGPNet'], default=['FSGPNet'])
    parser.add_argument('--dataset_names', nargs='+', choices=DATASETS, default=['NUDT-SIRST'])
    parser.add_argument('--dataset_dir', default='./datasets')
    parser.add_argument('--batchSize', type=int, default=8)
    parser.add_argument('--save', default='./log')
    parser.add_argument('--resume', nargs='+')
    parser.add_argument('--pretrained', nargs='+')
    parser.add_argument('--nEpochs', type=int, default=400)
    parser.add_argument('--optimizer_name', choices=['Adam', 'Adagrad', 'SGD'], default='Adam')
    parser.add_argument('--optimizer_settings', type=json.loads, default={'lr': 0.001}, help='JSON object')
    parser.add_argument('--scheduler_name', choices=['CosineAnnealingLR', 'MultiStepLR'], default='CosineAnnealingLR')
    parser.add_argument('--scheduler_settings', type=json.loads, default=None, help='JSON object')
    parser.add_argument('--loss_name', choices=['SoftIoU', 'softiou'], default='SoftIoU')
    parser.add_argument('--img_norm_cfg', type=json.loads, default=None, help='JSON mean/std object')
    parser.add_argument('--img_norm_cfg_mean', type=float)
    parser.add_argument('--img_norm_cfg_std', type=float)
    parser.add_argument('--threads', type=int, default=0)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--intervals', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--amp', action='store_true', help='Compatibility option; FSGPNet falls back to FP32.')
    args = parser.parse_args(argv)
    if args.resume and args.pretrained:
        parser.error('Use either --resume or --pretrained.')
    if min(args.batchSize, args.nEpochs, args.intervals) < 1 or args.threads < 0:
        parser.error('Batch size, epochs and intervals must be positive; threads must be nonnegative.')
    if not 0 <= args.threshold <= 1:
        parser.error('--threshold must be in [0, 1]')
    if (args.img_norm_cfg_mean is None) != (args.img_norm_cfg_std is None):
        parser.error('Provide both normalization mean and std.')
    if args.img_norm_cfg_mean is not None:
        args.img_norm_cfg = {'mean': args.img_norm_cfg_mean, 'std': args.img_norm_cfg_std}
    if args.img_norm_cfg and args.img_norm_cfg.get('std', 0) <= 0:
        parser.error('Normalization std must be positive.')
    settings = {'min_lr': 1e-5} if args.scheduler_name == 'CosineAnnealingLR' else {'step': [100, 200, 300], 'gamma': 0.5}
    if args.scheduler_settings:
        settings.update(args.scheduler_settings)
    settings['epochs'] = args.nEpochs
    args.scheduler_settings = settings
    return args


def select_checkpoint(paths, dataset_name, args):
    if not paths:
        return None
    if len(paths) == 1 and len(args.dataset_names) == 1:
        return paths[0]
    matches = [p for p in paths if dataset_name in Path(p).parts]
    if len(matches) != 1:
        raise ValueError(f'Provide exactly one checkpoint under a {dataset_name} directory.')
    return matches[0]


def train(args, dataset_name, model_name):
    seed_pytorch(args.seed)
    device = resolve_device(args.device)
    # FP16 DCT/Gabor responses produced NaN loss in the checked environment.
    # Keep the original model math and the manuscript's FP32 training protocol.
    use_amp = False
    if args.amp:
        warnings.warn('FSGPNet DCT/Gabor training is unstable in FP16; using FP32.')
    train_set = load_dataset(dataset_name, args.dataset_dir, 'trainval', args.img_norm_cfg)
    test_set = load_dataset(dataset_name, args.dataset_dir, 'test', args.img_norm_cfg)
    loader = DataLoader(train_set, batch_size=args.batchSize, shuffle=True, num_workers=args.threads)
    model = Net(model_name, mode='train').to(device)
    optimizer, scheduler = get_optimizer(model, args.optimizer_name, args.scheduler_name,
                                         args.optimizer_settings, args.scheduler_settings)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    start_epoch, history, best_miou = 0, [], -1.0
    checkpoint_path = select_checkpoint(args.resume or args.pretrained, dataset_name, args)
    if checkpoint_path:
        checkpoint = load_checkpoint(model, checkpoint_path)
        if args.resume:
            saved_config = checkpoint.get('config', {})
            for key in ('nEpochs', 'optimizer_name', 'scheduler_name', 'scheduler_settings'):
                if key in saved_config and saved_config[key] != getattr(args, key):
                    raise ValueError(f'Resume requires the original {key}: {saved_config[key]!r}. '
                                     'Use --pretrained for a new training schedule.')
            start_epoch = int(checkpoint.get('epoch', 0))
            history = checkpoint.get('history', [])
            best_miou = float(checkpoint.get('best_mIoU', -1.0))
            if 'optimizer' in checkpoint and 'scheduler' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
                if 'scaler' in checkpoint:
                    scaler.load_state_dict(checkpoint['scaler'])
            else:
                warnings.warn('Legacy checkpoint has no optimizer/scheduler state; resume is approximate.')
                scheduler.last_epoch = start_epoch
            if 'rng_state' in checkpoint:
                state = checkpoint['rng_state']
                random.setstate(state['python'])
                np.random.set_state(state['numpy'])
                torch.set_rng_state(state['torch'])
                if device.type == 'cuda' and state.get('cuda'):
                    torch.cuda.set_rng_state_all(state['cuda'])
        print(f'Loaded {checkpoint_path}')
    if start_epoch >= args.nEpochs:
        raise ValueError(f'Checkpoint epoch {start_epoch} is already >= --nEpochs {args.nEpochs}.')
    output = Path(args.save) / dataset_name / model_name
    output.mkdir(parents=True, exist_ok=True)
    # Save the underlying model to preserve the original state_dict keys.
    runner = torch.nn.DataParallel(model) if device.type == 'cuda' and torch.cuda.device_count() > 1 and device.index is None else model
    config = vars(args).copy()
    (output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    print(f'Training {dataset_name}/{model_name} on {device}; {len(train_set)} training images')
    with (output / 'train.txt').open('a', encoding='utf-8') as log:
        for epoch in range(start_epoch, args.nEpochs):
            runner.train()
            losses = []
            for img, mask in loader:
                img, mask = img.to(device), mask.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    pred = runner(img)
                    loss = model.loss(pred, mask)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Non-finite training loss at epoch {epoch + 1}.')
                scaler.scale(loss).backward()
                if use_amp:
                    scaler.unscale_(optimizer)
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError(f'Non-finite training gradient at epoch {epoch + 1}.')
                scaler.step(optimizer)
                scaler.update()
                losses.append(loss.item())
            scheduler.step()
            row = {'epoch': epoch + 1, 'loss': float(np.mean(losses)), 'lr': optimizer.param_groups[0]['lr']}
            message = f"Epoch {epoch + 1}/{args.nEpochs}; loss: {row['loss']:.6f}; lr: {row['lr']:.8g}"
            is_best = False
            should_evaluate = (epoch + 1) % args.intervals == 0 or epoch + 1 == args.nEpochs
            if should_evaluate:
                metrics = evaluate(model, test_set, device, args.threshold, args.threads)
                row.update(metrics)
                message += '\n' + format_results(dataset_name, metrics)
                is_best = metrics['mIoU'] > best_miou
                best_miou = max(best_miou, metrics['mIoU'])
            history.append(row)
            state = {'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                     'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(), 'history': history,
                     'total_loss': [item['loss'] for item in history], 'best_mIoU': best_miou, 'config': config,
                     'rng_state': {'python': random.getstate(), 'numpy': np.random.get_state(),
                                   'torch': torch.get_rng_state(),
                                   'cuda': torch.cuda.get_rng_state_all() if device.type == 'cuda' else None}}
            torch.save(state, output / 'last.pth.tar')
            if should_evaluate:
                torch.save(state, output / f'{epoch + 1}.pth.tar')
            if is_best:
                torch.save(state, output / 'best.pth.tar')
            (output / 'metrics.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
            print(message)
            log.write(message + '\n')
            log.flush()


def main(argv=None):
    args = parse_args(argv)
    for dataset_name in args.dataset_names:
        for model_name in args.model_names:
            train(args, dataset_name, model_name)


if __name__ == '__main__':
    main()
