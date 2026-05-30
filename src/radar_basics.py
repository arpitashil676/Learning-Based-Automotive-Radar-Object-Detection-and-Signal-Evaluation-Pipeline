print("Radar Object Detection Project")

radar_frequency = 77e9

targets = [
    {"distance": 20, "velocity": 5},
    {"distance": 50, "velocity": -10},
    {"distance": 80, "velocity": 0},
]

print("Radar frequency:", radar_frequency, "Hz")
print("Targets:")

for target in targets:
    print(
        "Distance:",
        target["distance"],
        "meters | Velocity:",
        target["velocity"],
        "m/s"
    )