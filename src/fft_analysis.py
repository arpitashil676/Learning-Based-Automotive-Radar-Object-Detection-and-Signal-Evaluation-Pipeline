import numpy as np
import matplotlib.pyplot as plt


def calculate_beat_frequency(distance, slope, speed_of_light):
    beat_frequency = (2 * slope * distance) / speed_of_light
    return beat_frequency


def estimate_distance_from_frequency(peak_frequency, slope, speed_of_light):
    estimated_distance = (peak_frequency * speed_of_light) / (2 * slope)
    return estimated_distance


# Radar constants
speed_of_light = 3e8
bandwidth = 150e6
chirp_time = 40e-6
slope = bandwidth / chirp_time

# Signal settings
sampling_rate = 100e6
duration = 40e-6

t = np.arange(0, duration, 1 / sampling_rate)

# Target
test_distances = [20, 50, 80]

errors = []
results = []

for true_distance in test_distances:
    beat_frequency = calculate_beat_frequency(
        true_distance,
        slope,
        speed_of_light
    )

    signal = np.sin(2 * np.pi * beat_frequency * t)

    fft_result = np.fft.fft(signal)
    fft_magnitude = np.abs(fft_result)

    frequency_axis = np.fft.fftfreq(len(t), 1 / sampling_rate)

    peak_index = np.argmax(fft_magnitude)
    peak_frequency = abs(frequency_axis[peak_index])

    estimated_distance = estimate_distance_from_frequency(
        peak_frequency,
        slope,
        speed_of_light
    )

    error = estimated_distance - true_distance
    errors.append(abs(error))

    results.append({
    "true_distance": true_distance,
    "beat_frequency": beat_frequency,
    "peak_frequency": peak_frequency,
    "estimated_distance": estimated_distance,
    "error": error
   })

    print("----------------------")
    print("True Distance:", true_distance, "m")
    print("Beat Frequency:", beat_frequency, "Hz")
    print("Detected Peak Frequency:", peak_frequency, "Hz")
    print("Estimated Distance:", estimated_distance, "m")
    print("Error:", error, "m")

mean_absolute_error = np.mean(errors)

print("\n======================")
print("Mean Absolute Error:", mean_absolute_error, "m")

print("\nResults Table")
print("True Distance | Estimated Distance | Error")

for row in results:
    print(
        row["true_distance"],
        "|",
        round(row["estimated_distance"], 4),
        "|",
        round(row["error"], 4)
    )