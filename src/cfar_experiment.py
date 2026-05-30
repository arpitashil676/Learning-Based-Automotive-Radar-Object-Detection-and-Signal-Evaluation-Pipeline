import numpy as np
from scipy.signal import find_peaks

print("CA-CFAR Parameter Experiment")

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
targets = [
    {"distance": 20, "amplitude": 2.0},
    {"distance": 50, "amplitude": 1.5},
    {"distance": 80, "amplitude": 1.2},
]

# Repeatable experiment
np.random.seed(42)

combined_signal = np.zeros_like(t)

for target in targets:
    distance = target["distance"]
    amplitude = target["amplitude"]

    beat_frequency = calculate_beat_frequency(
        distance,
        slope,
        speed_of_light
    )

    target_signal = amplitude * np.sin(2 * np.pi * beat_frequency * t)
    combined_signal += target_signal

noise_power = 1.0
print("Noise Power:", noise_power)
noise = noise_power * np.random.randn(len(t))
combined_signal += noise

print("Signal created")

fft_result = np.fft.fft(combined_signal)
fft_magnitude = np.abs(fft_result)

frequency_axis = np.fft.fftfreq(len(t), 1 / sampling_rate)

positive_indices = frequency_axis > 0
positive_frequencies = frequency_axis[positive_indices]
positive_magnitudes = fft_magnitude[positive_indices]

print("Maximum FFT Magnitude:", np.max(positive_magnitudes))

print("FFT completed")

threshold_factors = [1.5, 2, 2.5, 3]

print("\nCFAR Experiment Results")
print("-----------------------")

for threshold_factor in threshold_factors:

    detections = ca_cfar(
        positive_magnitudes,
        num_training_cells=20,
        num_guard_cells=8,
        threshold_factor=threshold_factor
    )

    print("Threshold factor", threshold_factor, "raw detections:", np.sum(detections))

    cfar_magnitudes = positive_magnitudes * detections

    peaks, _ = find_peaks(
        cfar_magnitudes,
        distance=10
    )

    detected_frequencies = positive_frequencies[peaks]

    detected_distances = (
    detected_frequencies * speed_of_light
    ) / (2 * slope)

    valid_distances = detected_distances[detected_distances < 120]
    
    print(
    "Threshold Factor:",
    threshold_factor,
    "| Raw Peaks:",
    len(peaks),
    "| Valid Targets:",
    len(valid_distances),
    "| Distances:",
    np.round(valid_distances, 2)
)