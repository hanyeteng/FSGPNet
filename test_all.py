"""Evaluate the matching pretrained checkpoint for each selected dataset."""
import argparse
from datetime import datetime
from pathlib import Path
from net import Net
from utils.runtime import DATASETS, evaluate, format_results, load_checkpoint, load_dataset, resolve_device


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_names', nargs='+', choices=['FSGPNet'], default=['FSGPNet'])
    parser.add_argument('--dataset_names', nargs='+', choices=DATASETS, default=list(DATASETS))
    parser.add_argument('--dataset_dir', default='./datasets')
    parser.add_argument('--checkpoint_dir', default='./checkpoints')
    parser.add_argument('--pth_dirs', nargs='+', default=None,
                        help='Optional DATASET/FSGPNet.pth.tar paths relative to checkpoint_dir.')
    parser.add_argument('--save_log', default='./log')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, choices=[1], default=1)
    parser.add_argument('--threads', type=int, default=0)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args(argv)
    if not 0 <= args.threshold <= 1 or args.threads < 0:
        parser.error('Threshold must be in [0, 1]; threads must be nonnegative.')
    return args


def main(argv=None):
    args = parse_args(argv)
    device = resolve_device(args.device)
    jobs = []
    for name in args.dataset_names:
        dataset = load_dataset(name, args.dataset_dir)
        for model_name in args.model_names:
            if args.pth_dirs:
                matches = [Path(p) for p in args.pth_dirs
                           if Path(p).parent.name == name and Path(p).name == f'{model_name}.pth.tar']
                if len(matches) != 1:
                    raise ValueError(f'Provide exactly one checkpoint for {name}/{model_name}.')
                relative = matches[0]
            else:
                relative = Path(name) / f'{model_name}.pth.tar'
            weight = Path(args.checkpoint_dir) / relative
            if not weight.is_file():
                raise FileNotFoundError(f'Missing checkpoint: {weight}')
            jobs.append((name, model_name, dataset, weight))
    output = Path(args.save_log)
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / f'test_{datetime.now():%Y%m%d_%H%M%S_%f}.txt'
    with log_path.open('w', encoding='utf-8') as log:
        for name, model_name, dataset, weight in jobs:
            print(f'Evaluating {name}/{model_name}: {len(dataset)} images on {device}', flush=True)
            model = Net(model_name, mode='test')
            load_checkpoint(model, weight)
            model.to(device)
            result = (f'Checkpoint: {weight}; device: {device}; threshold: {args.threshold}\n' +
                      format_results(name, evaluate(model, dataset, device, args.threshold, args.threads)))
            print(result)
            log.write(result + '\n\n')
            log.flush()
            del model
    print(f'Evaluation log saved to {log_path}')


if __name__ == '__main__':
    main()
