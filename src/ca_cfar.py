import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

np.random.seed(42)

def calculate_beat_frequency(distance, slope, speed_of_light):
    return (2 * slope * distance) / speed_of_light


def ca_cfar(signal, num_training_cells, num_guard_cells, threshold_factor):
    detections = np.zeros_like(signal, dtype=bool)

    start = num_training_cells + num_guard_cells
    end = len(signal) - num_training_cells - num_guard_cells

    for index in range(start, end):
        left_training = signal[
            index - num_guard_cells - num_training_cells:
            index - num_guard_cells
        ]

        right_training = signal[
            index + num_guard_cells + 1:
            index + num_guard_cells + num_training_cells + 1
        ]

        training_cells = np.concatenate((left_training, right_training))

        noise_level = np.mean(training_cells)
        threshold = threshold_factor * noise_level

        if signal[index] > threshold:
            detections[index] = True

    return detections


# Radar constants
speed_of_light = 3e8
bandwidth = 150e6
chirp_time = 40e-6
slope = bandwidth / chirp_time

# Signal settings
sampling_rate = 100e6
duration = 40e-6
t = np.arange(0, duration, 1 / sampling_rate)

# Targets
target_distances = [20, 50, 80]

combined_signal = np.zeros_like(t)

for distance in target_distances:
    beat_frequency = calculate_beat_frequency(
        distance,
        slope,
        speed_of_light
    )

    target_signal = np.sin(2 * np.pi * beat_frequency * t)
    combined_signal += target_signal

# Add noise
noise_power = 3.0
noise = noise_power * np.random.randn(len(t))
combined_signal += noise

# FFT
fft_result = np.fft.fft(combined_signal)
fft_magnitude = np.abs(fft_result)

frequency_axis = np.fft.fftfreq(len(t), 1 / sampling_rate)

positive_indices = frequency_axis > 0
positive_frequencies = frequency_axis[positive_indices]
positive_magnitudes = fft_magnitude[positive_indices]

# CA-CFAR
detections = ca_cfar(
    positive_magnitudes,
    num_training_cells=20,
    num_guard_cells=3,
    threshold_factor=2
)

cfar_magnitudes = positive_magnitudes * detections

peaks, _ = find_peaks(
    cfar_magnitudes,
    distance=10
)

detected_frequencies = positive_frequencies[peaks]

detected_distances = (
    detected_frequencies * speed_of_light
) / (2 * slope)

# Keep only reasonable automotive-range targets
valid_indices = detected_distances < 100
detected_distances = detected_distances[valid_indices]

print("CA-CFAR Detection Script")
print("True targets:", target_distances)
print("Noise power:", noise_power)
print("Detected targets:", len(detected_distances))

print("\nCA-CFAR Detected Targets")

for index, distance in enumerate(detected_distances, start=1):
    print("Target", index, "→", round(distance, 3), "m")

# Plot
plt.plot(positive_frequencies, positive_magnitudes)
plt.scatter(
    detected_frequencies[valid_indices],
    positive_magnitudes[peaks][valid_indices]
)

plt.title("CA-CFAR Target Detection")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid(True)
plt.show()