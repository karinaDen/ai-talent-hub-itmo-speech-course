import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
from torch.utils.data import DataLoader, Subset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import soundfile as sf

def _sf_load(path, frame_offset=0, num_frames=-1, normalize=True,
             channels_first=True, format=None, buffer_size=4096, backend=None):
    # unused args kept for API compatibility with torchaudio.load signature
    _ = normalize, format, buffer_size, backend
    data, sr = sf.read(str(path), dtype='float32', always_2d=True)
    if frame_offset:
        data = data[frame_offset:]
    if num_frames > 0:
        data = data[:num_frames]
    t = torch.from_numpy(data.T.copy() if channels_first else data.copy())
    return t, sr

torchaudio.load = _sf_load

sys.path.insert(0, os.path.dirname(__file__))
from melbanks import LogMelFilterBanks


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
CLASSES = ['yes', 'no']
MAX_AUDIO_LEN = 16000          # 1 second at 16 kHz
PLOT_DIR = os.path.join(os.path.dirname(__file__), 'plots')


# ─────────────────────────────────────────────
# Dataset helpers
# ─────────────────────────────────────────────
def get_loaders(data_root: str, batch_size: int, num_workers: int = 2):
    os.makedirs(data_root, exist_ok=True)
    loaders = {}
    for split in ('training', 'validation', 'testing'):
        ds = SPEECHCOMMANDS(data_root, download=True, subset=split)
        # get_metadata() 
        indices = [i for i in range(len(ds)) if ds.get_metadata(i)[2] in CLASSES]
        subset = Subset(ds, indices)
        loaders[split] = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=(split == 'training'),
            collate_fn=_collate,
            num_workers=num_workers,
            pin_memory=True,
        )
        print(f"  {split}: {len(subset)} samples")
    return loaders['training'], loaders['validation'], loaders['testing']


def _collate(batch):
    waveforms, labels = [], []
    for waveform, _sr, label, *_ in batch:
        n = waveform.shape[-1]
        if n < MAX_AUDIO_LEN:
            waveform = nn.functional.pad(waveform, (0, MAX_AUDIO_LEN - n))
        else:
            waveform = waveform[..., :MAX_AUDIO_LEN]
        waveforms.append(waveform)
        labels.append(float(CLASSES.index(label)))
    return torch.stack(waveforms), torch.tensor(labels)


# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────
class SpeechCNN(nn.Module):

    def __init__(self, n_mels: int = 80, groups: int = 1):
        super().__init__()
        if n_mels % groups != 0:
            raise ValueError(f"groups={groups} must divide n_mels={n_mels}")

        self.log_mel = LogMelFilterBanks(n_mels=n_mels)

        self.net = nn.Sequential(
            nn.Conv1d(n_mels, n_mels, kernel_size=3, padding=1, groups=groups),
            nn.BatchNorm1d(n_mels),
            nn.ReLU(),
            nn.Conv1d(n_mels, n_mels, kernel_size=3, padding=1, groups=groups),
            nn.BatchNorm1d(n_mels),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(n_mels, n_mels, kernel_size=3, padding=1, groups=groups),
            nn.BatchNorm1d(n_mels),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(n_mels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, time) or (batch, time)
        if x.dim() == 3:
            x = x.squeeze(1)                   # (batch, time)
        features = self.log_mel(x)             # (batch, n_mels, n_frames)
        pooled = self.net(features).squeeze(-1)  # (batch, n_mels)
        return self.classifier(pooled).squeeze(-1)  # (batch,)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_flops(self, input_shape=(1, MAX_AUDIO_LEN)) -> int:
        try:
            from thop import profile
            dummy = torch.randn(1, *input_shape)
            flops, _ = profile(self, inputs=(dummy,), verbose=False)
            return int(flops)
        except ImportError:
            return -1


# ─────────────────────────────────────────────
# Training / evaluation
# ─────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for waveforms, labels in loader:
        waveforms, labels = waveforms.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(waveforms), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for waveforms, labels in loader:
        waveforms, labels = waveforms.to(device), labels.to(device)
        preds = (torch.sigmoid(model(waveforms)) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


def run_experiment(config: dict, train_loader, val_loader, test_loader,
                   device: str, epochs: int):
    n_mels = config['n_mels']
    groups = config['groups']

    model = SpeechCNN(n_mels=n_mels, groups=groups).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    params = model.count_parameters()
    flops = model.count_flops()
    label = f"n_mels={n_mels}, groups={groups}"
    print(f"\n{'─'*55}")
    print(f"  {label} | params={params:,} | flops={flops:,}")
    print(f"{'─'*55}")

    losses, val_accs, epoch_times = [], [], []

    for epoch in range(epochs):
        t0 = time.time()
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        elapsed = time.time() - t0
        val_acc = evaluate(model, val_loader, device)
        losses.append(loss)
        val_accs.append(val_acc)
        epoch_times.append(elapsed)
        print(f"  ep {epoch+1:02d}/{epochs} | loss={loss:.4f} | val_acc={val_acc:.4f} | {elapsed:.1f}s")

    test_acc = evaluate(model, test_loader, device)
    print(f"  → test accuracy: {test_acc:.4f}")

    return {
        'label': label,
        'n_mels': n_mels,
        'groups': groups,
        'params': params,
        'flops': flops,
        'losses': losses,
        'val_accs': val_accs,
        'epoch_times': epoch_times,
        'test_acc': test_acc,
    }


# ─────────────────────────────────────────────
# Verification: LogMelFilterBanks ≈ MelSpectrogram
# ─────────────────────────────────────────────
def verify_and_plot(wav_path: str):
    """Assert numerical equivalence with torchaudio MelSpectrogram and save comparison plot."""
    signal, _ = torchaudio.load(wav_path)

    mel_transform = torchaudio.transforms.MelSpectrogram(hop_length=160, n_mels=80)
    melspec = mel_transform(signal)                    # (1, 80, T)

    log_mel_layer = LogMelFilterBanks()
    logmelbanks = log_mel_layer(signal)                # (1, 80, T)

    reference = torch.log(melspec + 1e-6)

    assert reference.shape == logmelbanks.shape, (
        f"Shape mismatch: {reference.shape} vs {logmelbanks.shape}"
    )
    assert torch.allclose(reference, logmelbanks), (
        f"Max diff: {(reference - logmelbanks).abs().max():.2e}"
    )
    print("✓  LogMelFilterBanks matches torchaudio.transforms.MelSpectrogram")

    # --- plot ---
    os.makedirs(PLOT_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].imshow(reference[0].numpy(), aspect='auto', origin='lower')
    axes[0].set_title('log(MelSpectrogram + 1e-6)  [torchaudio]')
    axes[0].set_xlabel('Frame')
    axes[0].set_ylabel('Mel bin')

    axes[1].imshow(logmelbanks[0].detach().numpy(), aspect='auto', origin='lower')
    axes[1].set_title('LogMelFilterBanks  [our impl]')
    axes[1].set_xlabel('Frame')
    axes[1].set_ylabel('Mel bin')

    plt.tight_layout()
    out = os.path.join(PLOT_DIR, 'melbanks_comparison.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────
def plot_loss_curves(results, title, filename):
    plt.figure(figsize=(10, 4))
    for r in results:
        plt.plot(r['losses'], label=r['label'])
    plt.xlabel('Epoch')
    plt.ylabel('Train Loss (BCE)')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_single(x_vals, y_vals, xlabel, ylabel, title, filename):
    plt.figure(figsize=(7, 4))
    plt.plot(x_vals, y_vals, 'o-', linewidth=2, markersize=8)
    for xv, yv in zip(x_vals, y_vals):
        plt.annotate(f'{yv:.4g}', (xv, yv), textcoords='offset points', xytext=(0, 8), ha='center')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_groups_summary(groups_results):
    groups = [r['groups'] for r in groups_results]
    avg_times = [sum(r['epoch_times']) / len(r['epoch_times']) for r in groups_results]
    params = [r['params'] for r in groups_results]
    flops = [r['flops'] for r in groups_results]
    test_accs = [r['test_acc'] for r in groups_results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle('Effect of groups parameter (n_mels=80)', fontsize=14)

    for ax, y, ylabel, title in zip(
        axes.flat,
        [avg_times, params, flops, test_accs],
        ['Avg epoch time (s)', 'Parameters', 'FLOPs', 'Test Accuracy'],
        ['Epoch Training Time', 'Model Parameters', 'FLOPs', 'Test Accuracy'],
    ):
        if all(v > 0 for v in y):
            ax.plot(groups, y, 'o-', linewidth=2, markersize=8)
            for xv, yv in zip(groups, y):
                ax.annotate(f'{yv:.4g}', (xv, yv), textcoords='offset points', xytext=(0, 8), ha='center')
        ax.set_xlabel('groups')
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, 'groups_summary.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--data-root', type=str, default='./data')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    os.makedirs(PLOT_DIR, exist_ok=True)

    # ── 0. Verify LogMelFilterBanks ──────────────────────────────────────
    print("\n[0] Verifying LogMelFilterBanks ...")
    sample_wav = None
    for root, _dirs, files in os.walk(args.data_root):
        for f in files:
            if f.endswith('.wav'):
                sample_wav = os.path.join(root, f)
                break
        if sample_wav:
            break

    # ── 1. Load data ─────────────────────────────────────────────────────
    print("\n[1] Loading SPEECHCOMMANDS (yes/no only) ...")
    train_loader, val_loader, test_loader = get_loaders(
        args.data_root, args.batch_size, args.workers
    )

    if sample_wav is None:
        for root, _dirs, files in os.walk(args.data_root):
            for f in files:
                if f.endswith('.wav'):
                    sample_wav = os.path.join(root, f)
                    break
            if sample_wav:
                break

    if sample_wav:
        verify_and_plot(sample_wav)

    # ── 2. Experiment 1: vary n_mels, groups=1 ───────────────────────────
    print("\n[2] Experiment 1 — n_mels ∈ [20, 40, 80], groups=1")
    nmels_results = []
    for n_mels in [20, 40, 80]:
        r = run_experiment(
            {'n_mels': n_mels, 'groups': 1},
            train_loader, val_loader, test_loader,
            device, args.epochs,
        )
        nmels_results.append(r)

    plot_loss_curves(nmels_results, 'Train Loss — effect of n_mels (groups=1)', 'nmels_loss.png')
    plot_single(
        [r['n_mels'] for r in nmels_results],
        [r['test_acc'] for r in nmels_results],
        'n_mels', 'Test Accuracy', 'n_mels vs Test Accuracy', 'nmels_accuracy.png',
    )
    plot_single(
        [r['n_mels'] for r in nmels_results],
        [r['params'] for r in nmels_results],
        'n_mels', 'Parameters', 'n_mels vs Parameters', 'nmels_params.png',
    )

    # ── 3. Experiment 2: vary groups, n_mels=80 ──────────────────────────
    print("\n[3] Experiment 2 — groups ∈ [1, 2, 4, 8, 16], n_mels=80")
    groups_results = []
    for groups in [1, 2, 4, 8, 16]:
        r = run_experiment(
            {'n_mels': 80, 'groups': groups},
            train_loader, val_loader, test_loader,
            device, args.epochs,
        )
        groups_results.append(r)

    plot_loss_curves(groups_results, 'Train Loss — effect of groups (n_mels=80)', 'groups_loss.png')
    plot_groups_summary(groups_results)

    # ── 4. Summary table ─────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print(f"{'Config':<30} {'Params':>10} {'FLOPs':>12} {'Test Acc':>10}")
    print("═" * 60)
    for r in nmels_results + groups_results[1:]:  # skip groups=1 duplicate
        print(f"  {r['label']:<28} {r['params']:>10,} {r['flops']:>12,} {r['test_acc']:>10.4f}")
    print("═" * 60)
    print(f"\nAll plots saved to: {PLOT_DIR}")


if __name__ == '__main__':
    main()
