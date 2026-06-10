import numpy as np
import os

print("Radar Dataset Generation")

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
range_resolution = speed_of_light / (2 * bandwidth)

fast_time = np.linspace(0, chirp_time, num_samples)
slow_time = np.arange(num_chirps) * chirp_time

doppler_frequencies = np.fft.fftfreq(num_chirps, chirp_time)
doppler_frequencies_shifted = np.fft.fftshift(doppler_frequencies)
velocity_axis = (doppler_frequencies_shifted * wavelength) / 2

# Dataset parameters
num_samples_dataset = 500
max_targets_per_scene = 3
noise_power_range = (0.05, 0.3)

np.random.seed(0)

os.makedirs("data", exist_ok=True)

rdmaps = []
labels = []

print("Generating", num_samples_dataset, "radar scenes...")

for scene_index in range(num_samples_dataset):

    num_targets = np.random.randint(1, max_targets_per_scene + 1)

    scene_targets = []
    for _ in range(num_targets):
        target_range = np.random.uniform(5, 90)
        target_velocity = np.random.uniform(-20, 20)
        amplitude = np.random.uniform(0.8, 2.0)
        scene_targets.append((target_range, target_velocity, amplitude))

    radar_data = np.zeros((num_chirps, num_samples), dtype=complex)

    for target_range, target_velocity, amplitude in scene_targets:
        doppler_freq = (2 * target_velocity) / wavelength
        for chirp_index in range(num_chirps):
            range_phase = (
                2 * np.pi * slope
                * (2 * target_range / speed_of_light)
                * fast_time
            )
            doppler_phase = 2 * np.pi * doppler_freq * slow_time[chirp_index]
            radar_data[chirp_index, :] += amplitude * np.exp(1j * (range_phase + doppler_phase))

    noise_power = np.random.uniform(*noise_power_range)
    radar_data += noise_power * (
        np.random.randn(num_chirps, num_samples)
        + 1j * np.random.randn(num_chirps, num_samples)
    )

    range_fft = np.fft.fft(radar_data, axis=1)
    positive_range_fft = range_fft[:, :num_samples // 2]

    doppler_fft = np.fft.fft(positive_range_fft, axis=0)
    doppler_magnitude = np.abs(doppler_fft)
    rdmap = np.fft.fftshift(doppler_magnitude, axes=0)

    rdmaps.append(rdmap)

    scene_label = []
    for target_range, target_velocity, _ in scene_targets:
        range_bin = int(target_range / range_resolution)
        range_bin = min(range_bin, num_samples // 2 - 1)

        velocity_diffs = np.abs(velocity_axis - target_velocity)
        doppler_bin = int(np.argmin(velocity_diffs))

        scene_label.append((range_bin, doppler_bin, target_range, target_velocity))

    labels.append(scene_label)

    if (scene_index + 1) % 100 == 0:
        print(f"  Generated {scene_index + 1} / {num_samples_dataset} scenes")

rdmaps_array = np.array(rdmaps, dtype=np.float32)

# Save dataset
np.save("data/rdmaps.npy", rdmaps_array)
print("\nDataset saved:")
print("  data/rdmaps.npy — shape:", rdmaps_array.shape)

# Save labels as structured text
with open("data/labels.txt", "w") as f:
    for scene_index, scene_label in enumerate(labels):
        for range_bin, doppler_bin, true_range, true_velocity in scene_label:
            f.write(
                f"{scene_index},{range_bin},{doppler_bin},{true_range:.4f},{true_velocity:.4f}\n"
            )

print("  data/labels.txt — target annotations (scene_idx,range_bin,doppler_bin,range_m,velocity_ms)")
print("\nDataset generation complete.")
print("Total scenes:", num_samples_dataset)
print("RD map shape per scene:", rdmaps_array.shape[1:])
print("Range resolution:", round(range_resolution, 4), "m")
print("Velocity resolution:", round(abs(velocity_axis[1] - velocity_axis[0]), 4), "m/s")
