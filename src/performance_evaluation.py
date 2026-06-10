import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import label
import os

print("Performance Evaluation")

# ── Radar constants ───────────────────────────────────────────────────────────

speed_of_light = 3e8
carrier_frequency = 77e9
wavelength = speed_of_light / carrier_frequency
bandwidth = 150e6
chirp_time = 40e-6
slope = bandwidth / chirp_time

num_chirps = 64
num_samples = 256
range_resolution = speed_of_light / (2 * bandwidth)

fast_time = np.linspace(0, chirp_time, num_samples)
slow_time = np.arange(num_chirps) * chirp_time

doppler_frequencies = np.fft.fftfreq(num_chirps, chirp_time)
doppler_frequencies_shifted = np.fft.fftshift(doppler_frequencies)
velocity_axis = (doppler_frequencies_shifted * wavelength) / 2
range_axis = np.arange(num_samples // 2) * range_resolution

# ── Helpers ───────────────────────────────────────────────────────────────────

def generate_scene(targets, noise_power, seed=None):
    if seed is not None:
        np.random.seed(seed)
    radar_data = np.zeros((num_chirps, num_samples), dtype=complex)
    for t in targets:
        doppler_freq = (2 * t["velocity"]) / wavelength
        for chirp_index in range(num_chirps):
            range_phase = (
                2 * np.pi * slope
                * (2 * t["range"] / speed_of_light)
                * fast_time
            )
            doppler_phase = 2 * np.pi * doppler_freq * slow_time[chirp_index]
            radar_data[chirp_index, :] += t["amplitude"] * np.exp(
                1j * (range_phase + doppler_phase)
            )
    radar_data += noise_power * (
        np.random.randn(num_chirps, num_samples)
        + 1j * np.random.randn(num_chirps, num_samples)
    )
    return radar_data


def compute_rdmap(radar_data):
    range_fft = np.fft.fft(radar_data, axis=1)
    positive_range_fft = range_fft[:, :num_samples // 2]
    doppler_fft = np.fft.fft(positive_range_fft, axis=0)
    doppler_magnitude = np.abs(doppler_fft)
    return np.fft.fftshift(doppler_magnitude, axes=0)


def detect_targets(rdmap, threshold_factor=4.0):
    noise_floor = np.mean(rdmap)
    threshold = threshold_factor * noise_floor
    detection_mask = rdmap > threshold
    labeled_array, num_features = label(detection_mask)
    detections = []
    for i in range(1, num_features + 1):
        pixels = np.argwhere(labeled_array == i)
        magnitudes = [rdmap[r, c] for r, c in pixels]
        peak = pixels[np.argmax(magnitudes)]
        doppler_idx, range_idx = peak
        detections.append({
            "range": range_axis[range_idx],
            "velocity": velocity_axis[doppler_idx],
        })
    return detections


def match_detections(true_targets, detections, range_tol=2.0, vel_tol=2.0):
    matched_true = set()
    matched_det = set()
    for det_idx, det in enumerate(detections):
        for true_idx, true in enumerate(true_targets):
            if true_idx in matched_true:
                continue
            if (abs(det["range"] - true["range"]) <= range_tol
                    and abs(det["velocity"] - true["velocity"]) <= vel_tol):
                matched_true.add(true_idx)
                matched_det.add(det_idx)
                break
    tp = len(matched_true)
    fp = len(detections) - len(matched_det)
    fn = len(true_targets) - len(matched_true)
    return tp, fp, fn


# ── Evaluation loop ───────────────────────────────────────────────────────────

np.random.seed(42)

noise_levels = [0.1, 0.3, 0.5, 1.0, 2.0, 3.0]
num_scenes_per_level = 50

results = {}

print(f"\n{'Noise':<8} {'Precision':<12} {'Recall':<10} {'F1':<10} {'Range MAE (m)':<16} {'Vel MAE (m/s)'}")

for noise_power in noise_levels:

    all_tp = all_fp = all_fn = 0
    range_errors = []
    vel_errors = []

    for scene_seed in range(num_scenes_per_level):
        num_targets = np.random.randint(1, 4)
        targets = []
        for _ in range(num_targets):
            targets.append({
                "range": np.random.uniform(10, 85),
                "velocity": np.random.uniform(-15, 15),
                "amplitude": np.random.uniform(0.9, 1.8),
            })

        radar_data = generate_scene(targets, noise_power, seed=scene_seed)
        rdmap = compute_rdmap(radar_data)
        detections = detect_targets(rdmap)

        tp, fp, fn = match_detections(targets, detections)
        all_tp += tp
        all_fp += fp
        all_fn += fn

        for det in detections:
            closest = min(targets, key=lambda t: abs(t["range"] - det["range"]))
            range_errors.append(abs(det["range"] - closest["range"]))
            vel_errors.append(abs(det["velocity"] - closest["velocity"]))

    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
    recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    range_mae = np.mean(range_errors) if range_errors else float("nan")
    vel_mae = np.mean(vel_errors) if vel_errors else float("nan")

    results[noise_power] = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "range_mae": range_mae,
        "vel_mae": vel_mae,
    }

    print(
        f"{noise_power:<8.1f} {precision:<12.3f} {recall:<10.3f} "
        f"{f1:<10.3f} {range_mae:<16.3f} {vel_mae:.3f}"
    )

# ── Plots ─────────────────────────────────────────────────────────────────────

os.makedirs("results", exist_ok=True)

noise_vals = list(results.keys())
precisions = [results[n]["precision"] for n in noise_vals]
recalls = [results[n]["recall"] for n in noise_vals]
f1s = [results[n]["f1"] for n in noise_vals]
range_maes = [results[n]["range_mae"] for n in noise_vals]
vel_maes = [results[n]["vel_mae"] for n in noise_vals]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(noise_vals, precisions, marker="o", label="Precision")
axes[0].plot(noise_vals, recalls, marker="s", label="Recall")
axes[0].plot(noise_vals, f1s, marker="^", label="F1 Score")
axes[0].set_xlabel("Noise Power")
axes[0].set_ylabel("Score")
axes[0].set_title("Detection Performance vs Noise")
axes[0].legend()
axes[0].grid(True)
axes[0].set_ylim(0, 1.05)

axes[1].plot(noise_vals, range_maes, marker="o", label="Range MAE (m)")
axes[1].plot(noise_vals, vel_maes, marker="s", label="Velocity MAE (m/s)")
axes[1].set_xlabel("Noise Power")
axes[1].set_ylabel("Mean Absolute Error")
axes[1].set_title("Estimation Accuracy vs Noise")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig("results/performance_evaluation.png", dpi=300)
print("\nPlot saved to results/performance_evaluation.png")
print("Performance evaluation complete.")
