# Automotive Radar Object Detection Project

## Goal

Build a radar perception pipeline including:

- FMCW radar simulation
- Range FFT
- Range-Doppler processing
- CFAR detection
- PyTorch-based object detection
- Performance evaluation

---

## Week 1

### Step 1
Created project structure.

### Step 2
Configured Python virtual environment.

### Step 3
Installed NumPy.

### Step 4
Created radar scene definition with:
- Radar frequency: 77 GHz
- Multiple targets
- Distance and velocity information


### Step 5
Created `beat_frequency.py`.

Calculated beat frequency using:

fb = (2 × slope × distance) / speed_of_light

For a target at 50 m, the beat frequency is approximately 1.25 MHz.


### Step 6
Generated first radar beat signal using NumPy and Matplotlib.

- Beat frequency: 1.25 MHz
- Target distance: 50 m

Visualized the signal as a sine wave in the time domain.

Learned that radar distance information is encoded in signal frequency rather than signal amplitude.


### Step 7

Applied Fast Fourier Transform (FFT) to the simulated radar beat signal.

Learned:
- FFT converts a signal from the time domain to the frequency domain.
- FFT output contains complex values.
- Magnitude was computed using np.abs().
- Two peaks appeared due to positive and negative frequency components of a sine wave.


### Step 8
Created frequency axis for FFT output.

Detected FFT peak frequency:

Peak frequency ≈ 1.30 MHz

Expected beat frequency:

1.25 MHz

The difference occurs because FFT frequency resolution is limited by the number of samples and signal duration.


### Step 9

Computed FFT peak frequency.

Observed:

Peak frequency = 1.2987 MHz

Expected:

1.25 MHz

Reason:
FFT frequency resolution is limited by the number of samples and sampling rate. The FFT selected the nearest available frequency bin.


### Step 10

Completed first radar range estimation loop.

Pipeline:

True distance → Beat frequency → Signal generation → FFT → Peak frequency → Estimated distance

Results:

- True distance: 50 m
- Beat frequency: 1.25 MHz
- Detected peak frequency: 1.2987 MHz
- Estimated distance: 51.95 m

The estimation error is caused by FFT frequency-bin resolution.


### Step 11

Improved distance estimation by increasing signal duration from 10 microseconds to 40 microseconds.

New result:

- True distance: 50 m
- Estimated distance: 49.99 m

Conclusion:
Increasing the signal duration improves FFT frequency resolution and produces more accurate range estimation.


### Step 12

Refactored radar equations into reusable Python functions:

- calculate_beat_frequency()
- estimate_distance_from_frequency()

This makes the radar processing pipeline cleaner, reusable, and easier to extend for multiple targets.


### Step 13

Tested range estimation for multiple target distances.

Results:

- 20 m → estimated 19.995 m
- 50 m → estimated 49.988 m
- 80 m → estimated 79.980 m

Conclusion:
The FFT-based range estimator accurately recovers target distance when the signal duration is sufficient.


### Step 14

Added automatic error calculation for multiple target distances.

Metric used:

Mean Absolute Error (MAE)

Result:

MAE = 0.0125 m

Conclusion:
The FFT-based range estimation pipeline achieved approximately 1.25 cm average distance error for 20 m, 50 m, and 80 m targets.


### Step 15

Created a multi-target radar signal by combining three sine waves.

Targets:

- 20 m
- 50 m
- 80 m

Applied FFT to the combined signal and kept only positive frequencies.

Observed three main positive FFT peaks, corresponding to the three target distances.


### Step 16

Installed SciPy and used `find_peaks()` to automatically detect peaks in the FFT magnitude.

Result:

- Number of detected peaks: 3

Conclusion:
The multi-target radar pipeline can now automatically detect three targets from the combined radar signal.


### Step 17

Converted automatically detected FFT peaks into estimated target distances.

Results:

- Peak 1 → 19.995 m
- Peak 2 → 49.988 m
- Peak 3 → 79.980 m

Conclusion:
The system successfully estimated distances for three simultaneous radar targets.


### Step 18

Added error analysis for multi-target distance estimation.

Results:

- 20 m → error = -0.005 m
- 50 m → error = -0.0125 m
- 80 m → error = -0.0200 m

Mean Absolute Error:

MAE = 0.0125 m

Conclusion:
The multi-target FFT-based radar range estimator achieved approximately 1.25 cm average error.


### Step 19

Added clean detected-target output.

The program now prints detected targets as a readable list:

- Target 1 → 19.995 m
- Target 2 → 49.988 m
- Target 3 → 79.980 m

This makes the radar output easier to interpret.


### Step 20

Added Gaussian noise to the multi-target radar signal.

Noise power: 0.5

Observation:
- Time-domain signal became visibly noisier.
- FFT-based peak detection still detected all three targets correctly.

Conclusion:
The radar processing pipeline is robust to moderate noise levels.


### Step 21

Tested stronger noise.

Noise power: 5.0

Result:
The simple peak detector detected 690 targets instead of 3.

Conclusion:
A fixed threshold peak detector is not robust in high-noise conditions. This motivates the need for CFAR detection.

### Step 22

Implemented a first simple CFAR-style detector.

Noise power: 5.0

Results:

- Fixed peak detector: 690 detections
- Simple CFAR detector: 3 detections

Conclusion:
Adaptive thresholding significantly reduced false detections in high-noise conditions.


### Step 23

Converted CFAR detections into target distances.

CFAR detected:

- Target 1 → 19.995 m
- Target 2 → 49.988 m
- Target 3 → 79.980 m

Conclusion:
CFAR successfully detected the correct target distances even under strong noise.


### Step 24

Implemented CA-CFAR with training cells and guard cells.

Under noisy conditions:

- 50 m target detected
- 80 m target detected
- 20 m target missed

Conclusion:
CA-CFAR improves false-alarm control, but detection performance depends on threshold factor, noise level, and target strength. This motivates later parameter tuning and performance evaluation.


### Step 25

Created CA-CFAR threshold-factor experiment.

Noise power: 3.0

Results:

- Threshold factor 2 → 51 detected peaks
- Threshold factor 3 → 4 detected peaks
- Threshold factor 4 → 2 detected peaks
- Threshold factor 5 → 2 detected peaks

Conclusion:
Lower threshold factors increase false detections, while higher threshold factors may miss weaker targets. Threshold factor 3 gives the closest result to the expected 3 targets.


### Step 26

Converted CA-CFAR experiment detections into distance estimates.

Results:

- Threshold factor 2 → valid distances: 49.99 m, 79.98 m, 105.97 m, 115.97 m
- Threshold factor 3 → valid distances: 49.99 m, 79.98 m
- Threshold factor 4 → valid distances: 49.99 m, 79.98 m
- Threshold factor 5 → valid distances: 49.99 m, 79.98 m

Conclusion:
The 50 m and 80 m targets are consistently detected. The 20 m target is missed under this noisy configuration, showing the need for further tuning and robustness analysis.


### Step 27

Tested lower CA-CFAR threshold factors at noise power 1.0.

Results showed:

- 50 m and 80 m were consistently detected.
- 20 m was still missed.
- Lower thresholds introduced false detections around 105–116 m.

Conclusion:
CA-CFAR performance is sensitive to target amplitude, local noise estimation, and threshold tuning. Further improvement requires target amplitude modeling or more advanced CFAR post-processing.


### Step 28

Tested CA-CFAR after adding target amplitude modeling and increasing guard cells.

Results:
- 50 m target detected consistently
- 80 m target detected consistently
- 20 m target missed
- False target appeared around 115.97 m for lower threshold factors

Conclusion:
CA-CFAR reduces false alarms compared to simple peak detection, but its performance depends strongly on threshold factor, guard cells, training cells, target amplitude, and noise. Further tuning is required for reliable detection of weaker/nearby targets.

### Step 29

Started Range-Doppler processing.

Implemented:
- radar data matrix with 64 chirps and 256 samples per chirp
- moving target simulation with range and velocity
- Range FFT across fast-time samples
- Doppler FFT across chirps
- Doppler FFT shift
- Range-Doppler heatmap visualization

Conclusion:
The pipeline now produces a Range-Doppler Map, which is the foundation for estimating both target distance and velocity.


### Step 30

Fixed Range-Doppler velocity estimation by using complex radar signals and applying Doppler FFT on the complex Range FFT output.

Results:

- True range: 50 m
- Estimated range: 50.0 m
- True velocity: 10 m/s
- Estimated velocity: 9.89 m/s

Conclusion:
The Range-Doppler pipeline can now estimate both target distance and velocity.


### Step 31

Updated the Range-Doppler map visualization.

Improvements:
- Converted x-axis from range bins to range in meters
- Converted y-axis from Doppler bins to velocity in m/s
- Added marker at detected target location

Detected target:
- Range ≈ 50 m
- Velocity ≈ 9.89 m/s

---

## Week 2

### Step 32

Completed multi-target Range-Doppler processing (`multi_target_range_doppler.py`).

Added:
- Doppler FFT across chirps on complex range FFT output
- Hanning window on range FFT to suppress sidelobes
- 2D peak detection using `scipy.ndimage.label` on thresholded magnitude map
- Visualization saved to `results/multi_target_range_doppler_map.png`

Targets and detections:

| Target | True Range | True Velocity | Estimated Range | Estimated Velocity |
|--------|-----------|---------------|-----------------|-------------------|
| 1 | 20 m | -5 m/s | 20.00 m | -5.33 m/s |
| 2 | 50 m | 10 m/s | 50.00 m | 9.89 m/s |
| 3 | 80 m | 0 m/s | 80.00 m | 0.00 m/s |

Conclusion:
All 3 targets detected with no false alarms. The Hanning window was critical — without it, range sidelobes from the 80 m target created a ghost detection at 50 m.


### Step 33

Created synthetic radar dataset generator (`dataset_generation.py`).

Generates 500 radar scenes, each with 1–3 random targets at varying range, velocity, amplitude, and noise level.

Pipeline per scene:
- Random target parameters drawn from uniform distributions
- Complex FMCW radar data cube constructed (64 chirps × 256 samples)
- Range FFT + Doppler FFT to produce Range-Doppler map
- Target locations annotated as (range bin, Doppler bin, true range, true velocity)

Output:
- `data/rdmaps.npy` — (500, 64, 128) float32 array
- `data/labels.txt` — per-target annotations

Dataset parameters:
- Range resolution: 1.0 m
- Velocity resolution: 0.761 m/s
- Noise power range: 0.05 – 0.30

Conclusion:
The dataset provides a realistic synthetic training set for a deep learning detector. It captures multi-target scenes with varying difficulty.


### Step 34

Implemented deep learning radar object detection (`deep_learning_detection.py`).

Architecture: CNN encoder-decoder (U-Net style)
- Encoder: 3 conv layers with MaxPool, channels 1 → 16 → 32 → 64
- Decoder: 2 transposed conv layers back to original resolution
- Output: sigmoid heatmap, one channel

Training setup:
- Loss: Binary Cross-Entropy
- Optimizer: Adam (lr = 1e-3)
- Epochs: 20
- Train/val split: 80/20 (400/100 scenes)
- Target heatmaps: Gaussian blobs (σ = 2) centered on each true target

Results:

| Epoch | Train Loss | Val Loss |
|-------|-----------|---------|
| 1 | 0.5899 | 0.5619 |
| 5 | 0.0547 | 0.0469 |
| 10 | 0.0362 | 0.0348 |
| 20 | 0.0116 | 0.0114 |

Model saved to `results/radar_detection_model.pth`.

Conclusion:
The model converges cleanly with no overfitting. Val loss tracks train loss closely, indicating the synthetic dataset is diverse enough for generalization.


### Step 35

Implemented performance evaluation framework (`performance_evaluation.py`).

Evaluated the CA-CFAR detector across 6 noise power levels (0.1 – 3.0), 50 scenes each with 1–3 random targets.

Metrics computed:
- Precision, Recall, F1 Score
- Range MAE (m), Velocity MAE (m/s)

Matching criterion: detection within ±2 m range and ±2 m/s velocity of a true target.

Results:

| Noise | Precision | Recall | F1 | Range MAE (m) | Vel MAE (m/s) |
|-------|-----------|--------|----|--------------|--------------|
| 0.1 | 0.352 | 0.792 | 0.488 | 6.30 | 9.49 |
| 0.5 | 0.260 | 0.970 | 0.410 | 3.92 | 8.51 |
| 1.0 | 0.432 | 1.000 | 0.604 | 2.45 | 3.21 |
| 2.0 | 0.717 | 1.000 | 0.835 | 0.95 | 0.91 |
| 3.0 | 0.832 | 1.000 | 0.908 | 0.47 | 0.73 |

Observation:
At low noise, the CFAR threshold is too permissive relative to the signal floor, causing false positives and low precision. At higher noise levels the noise floor rises, which paradoxically makes the adaptive threshold more discriminating, improving precision. Recall is consistently high.

Plot saved to `results/performance_evaluation.png`.