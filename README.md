# Radar Object Detection using FMCW Radar and CFAR

## Overview

This project implements a complete FMCW radar signal processing and object detection pipeline from scratch using Python, including classical signal processing, CFAR detection, Range-Doppler processing, and a deep learning detector.

## Features

- Beat frequency generation
- FFT-based range estimation
- Multi-target detection
- Noise modeling and error analysis
- Simple CFAR detection
- CA-CFAR with training and guard cells
- Parameter tuning experiments
- Range-Doppler Map (single and multi-target)
- Velocity estimation from Doppler shift
- Synthetic radar dataset generation
- Deep learning detection (CNN encoder-decoder)
- Performance evaluation across noise levels

## Project Structure

```text
src/
├── radar_basics.py                  # Radar constants and target definitions
├── beat_frequency.py                # Beat frequency calculation
├── signal_generation.py             # Radar beat signal synthesis
├── fft_analysis.py                  # FFT-based single-target range estimation
├── multi_target_detection.py        # Multi-target FFT peak detection
├── cfar_detection.py                # Simple adaptive CFAR detector
├── ca_cfar.py                       # CA-CFAR with training and guard cells
├── cfar_experiment.py               # CFAR threshold factor tuning study
├── range_doppler.py                 # Single-target Range-Doppler processing
├── multi_target_range_doppler.py    # Multi-target Range-Doppler with 2D detection
├── dataset_generation.py            # Synthetic radar dataset generation
├── deep_learning_detection.py       # CNN-based radar object detection (PyTorch)
└── performance_evaluation.py        # Precision, recall, F1, MAE across noise levels

data/
├── rdmaps.npy       # Generated Range-Doppler maps (500 scenes) — not tracked in git
└── labels.txt       # Target annotations (scene, range bin, Doppler bin, range, velocity)

results/
├── range_doppler_map.png              # Single-target Range-Doppler visualization
├── multi_target_range_doppler_map.png # Multi-target Range-Doppler with detections marked
├── performance_evaluation.png         # Precision/Recall/F1 and MAE vs noise plots
└── radar_detection_model.pth          # Trained CNN weights — not tracked in git

reports/
└── project_journal.md
```

## Results

| Metric | Value |
|---|---|
| Single-target range MAE | 0.0125 m |
| Single-target velocity error | ~1% |
| Multi-target detection (3 targets, no noise) | 3/3 detected, 0 false alarms |
| Deep learning val loss (20 epochs) | 0.011 |
| CFAR F1 at low noise (0.1) | 0.49 |
| CFAR F1 at high noise (3.0) | 0.91 |

## Pipeline

```
FMCW Signal → Range FFT → Doppler FFT → Range-Doppler Map
                                              ↓
                                    CA-CFAR / CNN Detector
                                              ↓
                                   Range + Velocity Estimates
                                              ↓
                                    Performance Evaluation
```

## Deep Learning Model

The CNN encoder-decoder (`deep_learning_detection.py`) treats detection as heatmap regression. It takes a normalized Range-Doppler map as input and outputs a probability heatmap where peaks mark target locations. Trained on 400 synthetic scenes, validated on 100.

To run:
```
python src/dataset_generation.py   # generate data first
python src/deep_learning_detection.py
```

## Dependencies

```
pip install numpy scipy matplotlib torch
```

## Author

Arpita Shil
M.Sc. Automotive Engineering
Technische Hochschule Ingolstadt
