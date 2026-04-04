# PolyPulse-PTT

Code repository for the paper:

> **Measuring multi-site pulse transit time with an AI-enabled mmWave radar**
> Jiangyifei Zhu\*, Kuang Yuan\*, Akarsh Prabhakara, Yunzhi Li, Gongwei Wang, Kelly Michaelsen, Justin Chan, Swarun Kumar
> *Nature Communications*, 2026.
> DOI: To be added

## Abstract

Pulse Transit Time (PTT) is a measure of arterial stiffness and a physiological marker associated with cardiovascular function, with an inverse relationship to diastolic blood pressure (DBP). We present an AI-enabled mmWave system for contactless multi-site PTT measurement using a single radar. By leveraging radar beamforming and deep learning algorithms our system simultaneously measures PTT and estimates diastolic blood pressure at multiple sites. The system was evaluated across three physiological pathways -- heart-to-radial artery, heart-to-carotid artery, and mastoid area-to-radial artery -- achieving correlation coefficients of 0.75--0.86 compared to contact-based reference sensors for measuring PTT. Furthermore, the system demonstrated correlation coefficients of 0.90--0.91 for estimating DBP, and achieved a mean error of -0.62--0.06 mmHg and standard deviation of 4.54--5.20 mmHg, meeting the FDA's AAMI guidelines for non-invasive blood pressure monitors. These results suggest that our proposed system has the potential to provide a non-invasive measure of cardiovascular health across multiple regions of the body.

This repository contains the complete signal-processing and machine-learning pipeline for estimating **Pulse Transit Time (PTT)** from multi-site radar and wearable sensor data.

---

## Repository Structure

```
polypulse-ptt/
├── code/
│   ├── preprocessing/       # MATLAB: sensor sync, bin selection, label extraction
│   │   ├── main.m           # Entry point for preprocessing
│   │   ├── wearable_sensor_sync.m
│   │   ├── radar_sensor_sync.m
│   │   ├── wearable_data_format.m
│   │   ├── radar_find_bin.m
│   │   └── utils/           # Signal processing utilities
│   ├── src/                 # Python: data preparation + ML inference
│   │   ├── prepare_data.py  # .mat → .npz sliding-window assembly
│   │   ├── test.py          # Model inference & PTT evaluation
│   │   ├── config/          # YAML configs per anatomical site
│   │   ├── data/            # PyTorch dataset + datamodule
│   │   └── models/          # Network architectures + training logic
│   ├── run                  # One-click reproducible pipeline script
│   └── LICENSE              # MIT License (code)
├── data/
│   └── LICENSE              # CC BY 4.0 (data)
├── LICENSE                  # Dual-license overview
└── .gitignore
```

## Requirements

**MATLAB** (R2021b or later) with Signal Processing Toolbox.

**Python 3.10+** with dependencies:

```bash
pip install -r code/src/requirements.txt
```

Key packages: PyTorch 2.6, PyTorch Lightning 2.5, SciPy 1.15, NumPy 2.2.

## Quick Start

### 1. Get the data

Download the example dataset and pre-trained checkpoint from Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19403782.svg)](https://doi.org/10.5281/zenodo.19403782)

Place files so the directory looks like:

```
data/
└── example/
    ├── original_sensor_data/
    │   ├── radar_data.mat
    │   └── wearable_data.mat
    └── processed/            # created by the pipeline
```

Place the checkpoint at `code/test_model.ckpt`.

### 2. Run the full pipeline

```bash
cd code
bash run
```

This executes three stages (~4 min total):

1. **Preprocessing (MATLAB)** — synchronize sensors, extract cardiac peaks, select radar bins
2. **Data preparation (Python)** — convert `.mat` → `.npz` with sliding windows
3. **Inference (Python)** — load checkpoint, evaluate PTT estimation, output `.csv` results

### 3. Run inference only (skip MATLAB)

If you already have preprocessed `.npz` files:

```bash
cd code
pip install -r src/requirements.txt
python3 src/test.py --checkpoint test_model.ckpt --leave_out_users example_user
```

## Configuration

The YAML files in `code/src/config/` define model architecture and hyperparameters for each anatomical site. The `joint.yaml` config is used by `test.py` for inference with the provided example data. The individual site configs (`head.yaml`, `heart.yaml`, etc.) contain `data_path` entries that point to the full training dataset (not included); update these paths if retraining on your own data.

## Pipeline Details

**Stage 1 — Preprocessing (MATLAB):** Synchronizes wearable IMU/ECG/PPG sensors with radar via cross-correlation. Detects cardiac features at four anatomical sites (head, heart, wrist, neck). Identifies radar range-angle bins carrying strong cardiac signatures.

**Stage 2 — Data Preparation (Python):** Pairs radar windows with wearable ground-truth peak labels. Applies 10-second sliding windows (80% overlap) at 500 Hz. Generates per-user `.npz` files for train/test splits.

**Stage 3 — Inference (Python):** Multi-site fusion network with cross-site attention. Estimates per-heartbeat PTT from radar-only features. Compares radar-derived PTT against wearable ground truth.

## Data Availability

The sensor data generated in this study have been deposited in Zenodo under accession code [10.5281/zenodo.19403782](https://doi.org/10.5281/zenodo.19403782). The raw radar and wearable sensor recordings are protected and not publicly available due to participant privacy under the IRB-approved study protocol. The processed example dataset (anonymized subset) is available via the Zenodo archive linked above, sufficient to reproduce the inference pipeline .

## Code Availability

All custom code used in this study is available at:

- **GitHub:** [https://github.com/ZJendex/polypulse-ptt](https://github.com/ZJendex/polypulse-ptt)
- **Zenodo:** [10.5281/zenodo.19403782](https://doi.org/10.5281/zenodo.19403782) (archived release)

## License

- **Code** (`code/`): [MIT License](code/LICENSE)
- **Data** (`data/`): [CC BY 4.0](data/LICENSE)

## Citation

If you use this code or data, please cite:

```bibtex
@article{zhu2026polypulse,
  title     = {Measuring multi-site pulse transit time with an AI-enabled mmWave radar},
  author    = {Zhu, Jiangyifei and Yuan, Kuang and Prabhakara, Akarsh and Li, Yunzhi and Wang, Gongwei and Michaelsen, Kelly and Chan, Justin and Kumar, Swarun},
  journal   = {Nature Communications},
  year      = {2026},
  doi       = {To be added}
}
```
