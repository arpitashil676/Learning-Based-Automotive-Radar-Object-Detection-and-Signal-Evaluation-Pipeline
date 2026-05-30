# Radar Object Detection using FMCW Radar and CFAR

## Overview

This project implements an FMCW radar signal processing pipeline from scratch using Python.

Features implemented:

- Beat frequency generation
- FFT-based range estimation
- Multi-target detection
- Noise modeling
- Error analysis
- Simple CFAR detection
- CA-CFAR implementation
- Parameter tuning experiments

## Project Structure

```text
src/
├── fft_analysis.py
├── multi_target_detection.py
├── cfar_detection.py
├── ca_cfar.py
└── cfar_experiment.py

reports/
└── project_journal.md
```

## Current Results

- Single-target range estimation
- Multi-target range estimation
- Mean Absolute Error ≈ 0.0125 m
- CFAR-based target detection under noisy conditions

## Future Work

- Range-Doppler Map
- Velocity Estimation
- Radar Dataset Generation
- Deep Learning for Radar Object Detection
- Performance Evaluation

## Author

Arpita Shil
M.Sc. Automotive Engineering
Technische Hochschule Ingolstadt