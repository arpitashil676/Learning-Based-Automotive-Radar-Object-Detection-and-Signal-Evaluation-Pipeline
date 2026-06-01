import numpy as np
import matplotlib.pyplot as plt


print("Range-Doppler Processing")

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

# Target definition
target_range = 50
target_velocity = 10

doppler_frequency = (
    2 * target_velocity
) / wavelength

# Radar data cube
radar_data = np.zeros(
    (num_chirps, num_samples),
    dtype=complex
)

for chirp_index in range(num_chirps):

    range_phase = 2 * np.pi * slope * (2 * target_range / speed_of_light) * fast_time

    doppler_phase = 2 * np.pi * doppler_frequency * slow_time[chirp_index]

    radar_data[chirp_index, :] = np.exp(
    1j * (range_phase + doppler_phase)
)

plt.plot(radar_data[0, :])

plt.title("Single Radar Chirp")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")

plt.grid(True)
plt.show()

# Range FFT
range_fft = np.fft.fft(radar_data, axis=1)

range_magnitude = np.abs(range_fft)

positive_range_fft = range_fft[:, :num_samples // 2]

print("Positive range FFT shape:", positive_range_fft.shape)

# Doppler FFT
doppler_fft = np.fft.fft(
    positive_range_fft,
    axis=0
)

doppler_magnitude = np.abs(doppler_fft)

doppler_magnitude_shifted = np.fft.fftshift(
    doppler_magnitude,
    axes=0
)

max_index = np.unravel_index(
    np.argmax(doppler_magnitude_shifted),
    doppler_magnitude_shifted.shape
)

doppler_bin = max_index[0]
range_bin = max_index[1]

print("Strongest target location:")
print("Doppler bin:", doppler_bin)
print("Range bin:", range_bin)

doppler_frequencies = np.fft.fftfreq(
    num_chirps,
    chirp_time
)

doppler_frequencies_shifted = np.fft.fftshift(
    doppler_frequencies
)

velocity_axis = (
    doppler_frequencies_shifted * wavelength
) / 2

estimated_velocity = velocity_axis[doppler_bin]

print("Estimated velocity:", estimated_velocity, "m/s")
range_resolution = speed_of_light / (2 * bandwidth)

estimated_range = range_bin * range_resolution

print("Range resolution:", range_resolution, "m")
print("Estimated range:", estimated_range, "m")

doppler_frequencies = np.fft.fftfreq(
    num_chirps,
    chirp_time
)

doppler_frequencies_shifted = np.fft.fftshift(
    doppler_frequencies
)

velocity_axis = (
    doppler_frequencies_shifted * wavelength
) / 2

estimated_velocity = velocity_axis[doppler_bin]

print("Estimated velocity:", estimated_velocity, "m/s")

print("Shifted Doppler magnitude shape:", doppler_magnitude_shifted.shape)

print("Doppler FFT shape:", doppler_fft.shape)
print("Doppler magnitude shape:", doppler_magnitude.shape)

print("Range FFT shape:", range_fft.shape)
print("Range magnitude shape:", range_magnitude.shape)

plt.plot(range_magnitude[0, :])

plt.title("Range FFT - First Chirp")
plt.xlabel("Range Bin")
plt.ylabel("Magnitude")

plt.grid(True)
plt.show()

print("Minimum value:", np.min(radar_data))
print("Maximum value:", np.max(radar_data))

range_axis = np.arange(num_samples // 2) * range_resolution

plt.imshow(
    doppler_magnitude_shifted,
    aspect="auto",
    origin="lower",
    extent=[
        range_axis[0],
        range_axis[-1],
        velocity_axis[0],
        velocity_axis[-1]
    ]
)

plt.scatter(
    estimated_range,
    estimated_velocity,
    marker="x",
    s=100
)

plt.title("Range-Doppler Map")
plt.xlabel("Range (m)")
plt.ylabel("Velocity (m/s)")

plt.colorbar(label="Magnitude")
plt.savefig("results/range_doppler_map.png", dpi=300)
plt.show()

plt.title("Range-Doppler Map")
plt.xlabel("Range Bin")
plt.ylabel("Doppler Bin")

plt.colorbar(label="Magnitude")
plt.show()

print("Wavelength:", wavelength, "m")
print("Slope:", slope)
print("Number of chirps:", num_chirps)
print("Number of samples per chirp:", num_samples)
print("Fast-time shape:", fast_time.shape)
print("Slow-time shape:", slow_time.shape)
print("Target range:", target_range, "m")
print("Target velocity:", target_velocity, "m/s")
print("Doppler frequency:", doppler_frequency, "Hz")
print("Radar data shape:", radar_data.shape)

