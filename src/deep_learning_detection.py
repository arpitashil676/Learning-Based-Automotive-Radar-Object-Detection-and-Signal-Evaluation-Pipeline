import numpy as np
import os

print("Deep Learning for Radar Object Detection")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    print("PyTorch version:", torch.__version__)
except ImportError:
    print("PyTorch not found. Install with: pip install torch")
    raise

# ── Dataset ──────────────────────────────────────────────────────────────────

class RadarDataset(Dataset):

    def __init__(self, rdmaps, heatmaps):
        self.rdmaps = torch.tensor(rdmaps, dtype=torch.float32).unsqueeze(1)
        self.heatmaps = torch.tensor(heatmaps, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.rdmaps)

    def __getitem__(self, index):
        return self.rdmaps[index], self.heatmaps[index]


def build_heatmaps(labels_path, num_scenes, map_shape, sigma=2):
    H, W = map_shape
    heatmaps = np.zeros((num_scenes, H, W), dtype=np.float32)

    rows = np.arange(H)[:, None]
    cols = np.arange(W)[None, :]

    with open(labels_path) as f:
        for line in f:
            parts = line.strip().split(",")
            scene_idx = int(parts[0])
            range_bin = int(parts[1])
            doppler_bin = int(parts[2])

            if scene_idx >= num_scenes:
                continue

            dist_sq = (rows - doppler_bin) ** 2 + (cols - range_bin) ** 2
            heatmaps[scene_idx] += np.exp(-dist_sq / (2 * sigma ** 2))

    heatmaps = np.clip(heatmaps, 0, 1)
    return heatmaps


# ── Model ─────────────────────────────────────────────────────────────────────

class RadarDetectionCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# ── Training ──────────────────────────────────────────────────────────────────

def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for rdmaps_batch, heatmaps_batch in loader:
        rdmaps_batch = rdmaps_batch.to(device)
        heatmaps_batch = heatmaps_batch.to(device)

        optimizer.zero_grad()
        predictions = model(rdmaps_batch)
        loss = criterion(predictions, heatmaps_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(rdmaps_batch)

    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for rdmaps_batch, heatmaps_batch in loader:
            rdmaps_batch = rdmaps_batch.to(device)
            heatmaps_batch = heatmaps_batch.to(device)
            predictions = model(rdmaps_batch)
            loss = criterion(predictions, heatmaps_batch)
            total_loss += loss.item() * len(rdmaps_batch)
    return total_loss / len(loader.dataset)


# ── Main ──────────────────────────────────────────────────────────────────────

rdmaps_path = "data/rdmaps.npy"
labels_path = "data/labels.txt"

if not os.path.exists(rdmaps_path):
    print("Dataset not found. Run dataset_generation.py first.")
    raise FileNotFoundError(rdmaps_path)

print("Loading dataset...")
rdmaps = np.load(rdmaps_path)
num_scenes = len(rdmaps)
map_shape = rdmaps.shape[1:]

print("RD maps shape:", rdmaps.shape)
print("Building target heatmaps...")

heatmaps = build_heatmaps(labels_path, num_scenes, map_shape)
print("Heatmaps shape:", heatmaps.shape)

# Normalize RD maps
rdmaps_max = rdmaps.max(axis=(1, 2), keepdims=True) + 1e-8
rdmaps_norm = rdmaps / rdmaps_max

# Train/val split
split = int(0.8 * num_scenes)
train_dataset = RadarDataset(rdmaps_norm[:split], heatmaps[:split])
val_dataset = RadarDataset(rdmaps_norm[split:], heatmaps[split:])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

print(f"Training samples: {len(train_dataset)} | Validation samples: {len(val_dataset)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

model = RadarDetectionCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCELoss()

num_params = sum(p.numel() for p in model.parameters())
print("Model parameters:", num_params)

print("\nTraining...")
print(f"{'Epoch':<8} {'Train Loss':<14} {'Val Loss':<12}")

num_epochs = 20
train_losses = []
val_losses = []

for epoch in range(1, num_epochs + 1):
    train_loss = train(model, train_loader, optimizer, criterion, device)
    val_loss = evaluate(model, val_loader, criterion, device)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"{epoch:<8} {train_loss:<14.6f} {val_loss:<12.6f}")

# Save model
os.makedirs("results", exist_ok=True)
torch.save(model.state_dict(), "results/radar_detection_model.pth")
print("\nModel saved to results/radar_detection_model.pth")

# Quick inference demo on one validation sample
model.eval()
sample_rdmap, sample_heatmap = val_dataset[0]
with torch.no_grad():
    pred = model(sample_rdmap.unsqueeze(0).to(device)).cpu().squeeze().numpy()

pred_peaks = np.argwhere(pred > 0.5)
print(f"\nInference demo — predicted {len(pred_peaks)} target pixel(s) above 0.5 threshold")
print("Final val loss:", round(val_losses[-1], 6))
