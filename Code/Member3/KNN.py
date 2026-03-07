from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main_pipeline import recognize_row_image


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Demo: run OCR pipeline from Member 3 entrypoint')
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--dataset-root', default='dataset', help='Training dataset root')
    parser.add_argument('--k', type=int, default=3, help='k for k-NN')
    return parser


if __name__ == '__main__':
    args = _build_parser().parse_args()
    text = recognize_row_image(args.image, dataset_root=args.dataset_root, k=args.k)
    print(text)
