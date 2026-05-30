import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Radar constants
speed_of_light = 3e8
bandwidth = 150e6
chirp_time = 40e-6
slope = bandwidth / chirp_time

def calculate_beat_frequency(distance, slope, speed_of_light):
    beat_frequency = (2 * slope * distance) / speed_of_light
    return beat_frequency

def estimate_distance_from_frequency(peak_frequency, slope, speed_of_light):
    estimated_distance = (peak_frequency * speed_of_light) / (2 * slope)
    return estimated_distance

# Signal settings
sampling_rate = 100e6
duration = 40e-6

t = np.arange(0, duration, 1 / sampling_rate)

combined_signal = np.zeros_like(t)
print("Combined signal shape:", combined_signal.shape)
print("Number of samples:", len(t))

# Multiple targets
target_distances = [20, 50, 80]

for distance in target_distances:
    beat_frequency = calculate_beat_frequency(
    distance,
    slope,
    speed_of_light
   )

    target_signal = np.sin(2 * np.pi * beat_frequency * t)

    combined_signal = combined_signal + target_signal

    print("Added target at", distance, "m")

noise_power = 5.0

noise = noise_power * np.random.randn(len(t))

combined_signal = combined_signal + noise

# Apply FFT
fft_result = np.fft.fft(combined_signal)
fft_magnitude = np.abs(fft_result)

frequency_axis = np.fft.fftfreq(len(t), 1 / sampling_rate)

# Keep only positive frequencies
positive_indices = frequency_axis > 0
positive_frequencies = frequency_axis[positive_indices]
positive_magnitudes = fft_magnitude[positive_indices]

peaks, _ = find_peaks(
    positive_magnitudes,
    height=100
)

print("Number of detected peaks:", len(peaks))

estimated_distances = []

for peak in peaks:
    peak_frequency = positive_frequencies[peak]

    estimated_distance = estimate_distance_from_frequency(
    peak_frequency,
    slope,
    speed_of_light
    )
    estimated_distances.append(estimated_distance)

    print("Peak frequency:", peak_frequency, "Hz")
    print("Estimated distance:", estimated_distance, "m")
    print("----------------")

print("\nDetected Targets")

for index, distance in enumerate(estimated_distances, start=1):
    print("Target", index, "→", round(distance, 3), "m")


print("Error Analysis")

errors = []

for true_distance, estimated_distance in zip(target_distances, estimated_distances):
    error = estimated_distance - true_distance
    errors.append(abs(error))

    print("True distance:", true_distance, "m")
    print("Estimated distance:", estimated_distance, "m")
    print("Error:", error, "m")
    print("----------------")


mean_absolute_error = np.mean(errors)

print("Mean Absolute Error:", mean_absolute_error, "m")

# Plot positive FFT
plt.plot(positive_frequencies, positive_magnitudes)

plt.title("Positive FFT of Multi-Target Radar Signal")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")

plt.grid(True)
plt.show()