import numpy as np
import matplotlib.pyplot as plt

# Time axis
t = np.linspace(0, 0.00001, 1000)

# Signal frequency (1.25 MHz from our 50 m target)
frequency = 1.25e6

# Generate signal
signal = np.sin(2 * np.pi * frequency * t)

# Plot
plt.plot(t, signal)

plt.title("Radar Beat Signal")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

plt.grid(True)

plt.show()