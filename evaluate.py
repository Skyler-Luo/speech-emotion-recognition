import argparse
import json
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.config import EMOTION_LABEL_MAP, NUM_CLASSES
from utils.dataset import EmotionDataset
from utils.utils import worker_init_fn
from utils.model_utils import build_model_from_checkpoint, evaluate_per_class


def main(args):
    device = (torch.device(f'cuda:{args.gpu_id}')
              if args.cuda and torch.cuda.is_available()
              else torch.device('cpu'))
    print(f"设备: {device}")

    dataset = EmotionDataset(
        dataset_dir=args.dataset_dir,
        feature_type=args.feature_type,
        max_length=args.max_length,
        n_mels=args.n_mels,
        hop_length=args.hop_length,
        normalize=True,
        random_offset=False,
        return_waveform=False,
        augmenter=None,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, worker_init_fn=worker_init_fn)

    print(f"加载检查点: {args.weights}")
    model, used_key = build_model_from_checkpoint(
        args.weights, device, num_classes=NUM_CLASSES)
    print(f"使用权重键: {used_key}")

    criterion = nn.CrossEntropyLoss()
    t0 = time.time()
    f1, result = evaluate_per_class(model, loader, criterion, device, verbose=True)
    elapsed = time.time() - t0
    print(f"\n评估完成，用时 {elapsed:.1f}s | "
          f"准确率 {result['accuracy']*100:.2f}% | F1 {f1*100:.2f}%")

    if args.save_results:
        from datetime import datetime
        out = {
            'model_weights':     args.weights,
            'dataset_dir':       args.dataset_dir,
            'accuracy':          float(result['accuracy']),
            'f1_macro':          float(result['f1_macro']),
            'precision_macro':   float(result['precision_macro']),
            'recall_macro':      float(result['recall_macro']),
            'loss':              float(result['loss']),
            'class_metrics': {
                k: {m: float(v) for m, v in v_dict.items() if m != 'samples'}
                for k, v_dict in result['class_metrics'].items()
            },
            'evaluation_time': elapsed,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        output_file = args.output_file or \
            f"results_{os.path.basename(args.weights).split('.')[0]}_{int(time.time())}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"结果已保存: {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='情感分类模型评估')
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--feature_type', type=str, default='mel_spectrogram')
    parser.add_argument('--n_mels', type=int, default=128)
    parser.add_argument('--hop_length', type=int, default=320)
    parser.add_argument('--max_length', type=int, default=3 * 32000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--no_cuda', dest='cuda', action='store_false')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--save_results', action='store_true', default=False)
    parser.add_argument('--output_file', type=str, default='')
    main(parser.parse_args())
