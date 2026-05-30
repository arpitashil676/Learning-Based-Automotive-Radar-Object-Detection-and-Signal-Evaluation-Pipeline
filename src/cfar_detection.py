import numpy as np
import matplotlib.pyplot as plt


# Radar constants
speed_of_light = 3e8
bandwidth = 150e6
chirp_time = 40e-6
slope = bandwidth / chirp_time

def calculate_beat_frequency(distance, slope, speed_of_light):
    beat_frequency = (2 * slope * distance) / speed_of_light
    return beat_frequency

def simple_cfar(signal, threshold_factor):
    noise_level = np.mean(signal)
    threshold = threshold_factor * noise_level

    detections = signal > threshold

    return detections, threshold

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

    target_signal = np.sin(
        2 * np.pi * beat_frequency * t
    )

    combined_signal += target_signal

    print("Added target at", distance, "m")

noise_power = 5.0

noise = noise_power * np.random.randn(len(t))

combined_signal = combined_signal + noise

print("Added noise with power:", noise_power)

fft_result = np.fft.fft(combined_signal)
fft_magnitude = np.abs(fft_result)

frequency_axis = np.fft.fftfreq(len(t), 1 / sampling_rate)

positive_indices = frequency_axis > 0
positive_frequencies = frequency_axis[positive_indices]
positive_magnitudes = fft_magnitude[positive_indices]

detections, threshold = simple_cfar(
    positive_magnitudes,
    threshold_factor=5
)

print("FFT completed")
print("Positive frequency bins:", len(positive_frequencies))

print("CFAR threshold:", threshold)
print("Number of CFAR detections:", np.sum(detections))

detected_frequencies = positive_frequencies[detections]

print("\nCFAR Detected Targets")

for index, frequency in enumerate(detected_frequencies, start=1):
    estimated_distance = (frequency * speed_of_light) / (2 * slope)

    print("Target", index, "→", round(estimated_distance, 3), "m")


plt.plot(positive_frequencies, positive_magnitudes)

plt.title("Noisy FFT Before CFAR")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")

plt.grid(True)
plt.show()


print("Combined signal shape:", combined_signal.shape)

print("Number of samples:", len(t))
print("Target distances:", target_distances)

print("CFAR Detection Script")
print("Radar slope:", slope)