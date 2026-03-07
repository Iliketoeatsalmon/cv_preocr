# Handwritten Capital Letter OCR (Classical CV + Manual k-NN)

This project recognizes English uppercase letters from an input image row.
Pipeline:

1. `segmentation.py` - split row image into character crops
2. `preprocessing.py` - normalize each char to 32x32 binary (ink=255)
3. `features.py` - extract 132-dim feature vector
4. `classifier.py` - manual k-NN predict
5. `main_pipeline.py` - run end-to-end recognition

## Project Structure

- `segmentation.py`
- `preprocessing.py`
- `features.py`
- `classifier.py`
- `evaluation.py`
- `main_pipeline.py`
- `dataset/`
- `interface_contract.md`

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Recognition

```bash
python3 main_pipeline.py --image dataset/writer1/A/A_001.png
```

With custom k and dataset root:

```bash
python3 main_pipeline.py --image path/to/row.jpg --dataset-root dataset --k 3
```

## Notes

- Labels are encoded as `A=0` to `Z=25`.
- Current sample dataset in this repo contains mostly `A-J` folders.
- If your image contains a single character, segmentation still works (returns one crop).
