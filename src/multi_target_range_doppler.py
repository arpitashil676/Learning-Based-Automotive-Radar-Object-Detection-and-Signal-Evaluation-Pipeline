import numpy as np
import matplotlib.pyplot as plt


print("Multi-Target Range-Doppler Processing")

# Radar constants
speed_of_light = 3e8
carrier_frequency = 77e9
wavelength = speed_of_light / carrier_frequency

bandwidth = 150e6
chirp_time = 40e-6
slope = bandwidth / chirp_time

# Range-Doppler settings
num_chirps = 64
num_samples = 256

# Time axes
fast_time = np.linspace(0, chirp_time, num_samples)
slow_time = np.arange(num_chirps) * chirp_time

# Multiple targets
targets = [
    {"range": 20, "velocity": -5, "amplitude": 1.5},
    {"range": 50, "velocity": 10, "amplitude": 1.2},
    {"range": 80, "velocity": 0, "amplitude": 1.0},
]

print("Targets:")

for target in targets:
    print(
        "Range:",
        target["range"],
        "m | Velocity:",
        target["velocity"],
        "m/s | Amplitude:",
        target["amplitude"]
    )

radar_data = np.zeros(
    (num_chirps, num_samples),
    dtype=complex
)

for target in targets:

    target_range = target["range"]
    target_velocity = target["velocity"]
    amplitude = target["amplitude"]

    doppler_frequency = (
        2 * target_velocity
    ) / wavelength

    for chirp_index in range(num_chirps):

        range_phase = (
            2 * np.pi
            * slope
            * (2 * target_range / speed_of_light)
            * fast_time
        )

        doppler_phase = (
            2 * np.pi
            * doppler_frequency
            * slow_time[chirp_index]
        )

        radar_data[chirp_index, :] += (
            amplitude
            * np.exp(
                1j * (range_phase + doppler_phase)
            )
        )

    print(
        "Added target:",
        target_range,
        "m,",
        target_velocity,
        "m/s"
    )


# Range FFT with Hanning window to suppress sidelobes
range_window = np.hanning(num_samples)
range_fft = np.fft.fft(radar_data * range_window, axis=1)
positive_range_fft = range_fft[:, :num_samples // 2]

print("Range FFT shape:", range_fft.shape)
print("Positive range FFT shape:", positive_range_fft.shape)

# Doppler FFT
doppler_fft = np.fft.fft(positive_range_fft, axis=0)
doppler_magnitude = np.abs(doppler_fft)
doppler_magnitude_shifted = np.fft.fftshift(doppler_magnitude, axes=0)

print("Doppler FFT shape:", doppler_fft.shape)

# Axes
range_resolution = speed_of_light / (2 * bandwidth)
range_axis = np.arange(num_samples // 2) * range_resolution

doppler_frequencies = np.fft.fftfreq(num_chirps, chirp_time)
doppler_frequencies_shifted = np.fft.fftshift(doppler_frequencies)
velocity_axis = (doppler_frequencies_shifted * wavelength) / 2

# 2D peak detection via row-wise CFAR threshold
noise_floor = np.mean(doppler_magnitude_shifted)
threshold = 8.0 * noise_floor

detection_mask = doppler_magnitude_shifted > threshold

# Find connected peaks using local maxima in 2D
from scipy.ndimage import label, maximum_position, find_objects

labeled_array, num_features = label(detection_mask)
print("Number of detections found:", num_features)

detected_targets = []

for feature_index in range(1, num_features + 1):
    feature_pixels = np.argwhere(labeled_array == feature_index)
    magnitudes = [doppler_magnitude_shifted[r, c] for r, c in feature_pixels]
    peak_local = feature_pixels[np.argmax(magnitudes)]
    doppler_idx, range_idx = peak_local
    est_range = range_axis[range_idx]
    est_velocity = velocity_axis[doppler_idx]
    detected_targets.append((est_range, est_velocity))

print("\nDetected Targets:")
print(f"{'#':<4} {'Range (m)':<12} {'Velocity (m/s)':<16}")
for i, (r, v) in enumerate(detected_targets, 1):
    print(f"{i:<4} {r:<12.2f} {v:<16.2f}")

print("\nTrue Targets:")
for t in targets:
    print(f"  Range: {t['range']} m | Velocity: {t['velocity']} m/s")

# Visualization
plt.figure(figsize=(10, 6))
plt.imshow(
    doppler_magnitude_shifted,
    aspect="auto",
    origin="lower",
    extent=[range_axis[0], range_axis[-1], velocity_axis[0], velocity_axis[-1]],
    cmap="jet"
)

for est_range, est_velocity in detected_targets:
    plt.scatter(est_range, est_velocity, color="white", marker="x", s=120, linewidths=2)

plt.title("Multi-Target Range-Doppler Map")
plt.xlabel("Range (m)")
plt.ylabel("Velocity (m/s)")
plt.colorbar(label="Magnitude")
plt.tight_layout()
plt.savefig("results/multi_target_range_doppler_map.png", dpi=300)
plt.show()

print("\nRadar setup complete")
print("Wavelength:", wavelength, "m")
print("Radar matrix size:", num_chirps, "x", num_samples)