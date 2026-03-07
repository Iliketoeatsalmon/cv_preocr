# Handwritten Capital Letter OCR (Classical CV + Manual k-NN)

This project recognizes English uppercase letters from an input image row.
Pipeline:

1. `segmentation.py` - split row image into character crops
2. `preprocessing.py` - normalize each char to 32x32 binary (ink=255)
3. `features.py` - extract 132-dim feature vector
4. `classifier.py` - manual k-NN predict
5. `main_pipeline.py` - run end-to-end recognition
6. `streamlit_app.py` - upload image and preview OCR results in a UI

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

## Run The Upload UI

```bash
streamlit run streamlit_app.py
```

The UI lets you:

- upload `.png`, `.jpg`, `.jpeg`, or `.bmp`
- preview the original image and detected character boxes
- view the recognized text
- inspect each segmented crop and its normalized 32x32 input

## Notes

- Labels are encoded as `A=0` to `Z=25`.
- Current sample dataset in this repo contains mostly `A-J` folders.
- If your image contains a single character, segmentation still works (returns one crop).
