from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features import extract_features
from preprocessing import preprocess_char


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Demo: extract feature vector for one character image')
    parser.add_argument('--image', required=True, help='Path to input image')
    return parser


if __name__ == '__main__':
    args = _build_parser().parse_args()
    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f'Cannot read image: {args.image}')

    norm = preprocess_char(img)
    feat = extract_features(norm)

    print(f'Feature length: {len(feat)}')
    print(f'First 10 values: {feat[:10]}')
