"""Evaluate one checkpoint on one or more datasets."""
import argparse
from net import Net
from utils.runtime import DATASETS, evaluate, format_results, load_checkpoint, load_dataset, resolve_device


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_name', choices=['FSGPNet'], default='FSGPNet')
    parser.add_argument('--dataset_names', nargs='+', choices=DATASETS, default=['IRSTD-1K'])
    parser.add_argument('--dataset_dir', default='./datasets')
    parser.add_argument('--weight_path', default='./checkpoints/IRSTD-1K/FSGPNet.pth.tar')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, choices=[1], default=1,
                        help='Full-image evaluation requires batch size 1.')
    parser.add_argument('--threads', type=int, default=0)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args(argv)
    if not 0 <= args.threshold <= 1 or args.threads < 0:
        parser.error('Threshold must be in [0, 1]; threads must be nonnegative.')
    return args


def main(argv=None):
    args = parse_args(argv)
    device = resolve_device(args.device)
    datasets = [(name, load_dataset(name, args.dataset_dir)) for name in args.dataset_names]
    model = Net(args.model_name, mode='test')
    load_checkpoint(model, args.weight_path)
    model.to(device)
    print(f'Model: {args.model_name}; device: {device}; checkpoint: {args.weight_path}')
    for name, dataset in datasets:
        print(format_results(name, evaluate(model, dataset, device, args.threshold, args.threads)))


if __name__ == '__main__':
    main()
