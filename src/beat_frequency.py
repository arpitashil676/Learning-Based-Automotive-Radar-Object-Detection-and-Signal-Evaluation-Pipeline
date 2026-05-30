def calculate_beat_frequency(distance):

    c = 3e8          # speed of light (m/s)

    bandwidth = 150e6   # 150 MHz

    chirp_time = 40e-6  # 40 microseconds

    slope = bandwidth / chirp_time

    beat_frequency = (2 * slope * distance) / c

    return beat_frequency


target_distance = 50

frequency = calculate_beat_frequency(target_distance)

print("Target distance:", target_distance, "m")

print("Beat frequency:", frequency, "Hz")