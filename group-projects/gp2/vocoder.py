"""
MelGAN-style neural vocoder: mel spectrogram -> waveform.

Generator:
  Input:  mel [B, 80, T_mel]
  Output: wav [B, 1, T_mel * 256]
  Upsampling ratios: 8 x 8 x 4 = 256 (matches TTS hop_length=256)

Discriminator:
  Multi-Scale Discriminator — 3 sub-discriminators at 1x, 2x, 4x downsampling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────
# Generator
# ─────────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """Residual stack with dilated convolutions (HiFi-GAN style)."""

    def __init__(self, channels: int, kernel_size: int = 3,
                 dilations: tuple = (1, 3, 9)):
        super().__init__()
        self.convs = nn.ModuleList()
        for d in dilations:
            p = d * (kernel_size - 1) // 2
            self.convs.append(nn.Sequential(
                nn.LeakyReLU(0.2),
                nn.Conv1d(channels, channels, kernel_size, dilation=d, padding=p),
                nn.LeakyReLU(0.2),
                nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2),
            ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = x + conv(x)
        return x


class Generator(nn.Module):
    """
    MelGAN-style generator with transposed-conv upsampling.

    Upsampling: 8 x 8 x 4 = 256  (matches hop_length=256 from TTS config)
    Channels:   512 -> 256 -> 128 -> 64
    """

    UPSAMPLE_RATIOS = [8, 8, 4]
    CHANNELS = [512, 256, 128, 64]

    def __init__(self, n_mels: int = 80):
        super().__init__()

        self.pre = nn.Conv1d(n_mels, self.CHANNELS[0], kernel_size=7, padding=3)

        self.ups = nn.ModuleList()
        self.res_blocks = nn.ModuleList()
        for r, ch_in, ch_out in zip(
            self.UPSAMPLE_RATIOS, self.CHANNELS, self.CHANNELS[1:]
        ):
            self.ups.append(
                nn.ConvTranspose1d(
                    ch_in, ch_out,
                    kernel_size=r * 2, stride=r, padding=r // 2,
                )
            )
            self.res_blocks.append(ResBlock(ch_out))

        self.post = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv1d(self.CHANNELS[-1], 1, kernel_size=7, padding=3),
            nn.Tanh(),
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.normal_(m.weight, 0.0, 0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = self.pre(mel)
        for up, res in zip(self.ups, self.res_blocks):
            x = F.leaky_relu(x, 0.2)
            x = up(x)
            x = res(x)
        return self.post(x)


# ─────────────────────────────────────────────────────────────────
# Multi-Scale Discriminator
# ─────────────────────────────────────────────────────────────────

class SubDiscriminator(nn.Module):
    """Single MelGAN-style discriminator operating on one audio scale."""

    def __init__(self):
        super().__init__()
        # Grouped convolutions reduce parameters while preserving capacity
        self.layers = nn.ModuleList([
            nn.Conv1d(1, 16, kernel_size=15, padding=7),
            nn.Conv1d(16, 64, kernel_size=41, stride=4, padding=20, groups=4),
            nn.Conv1d(64, 256, kernel_size=41, stride=4, padding=20, groups=16),
            nn.Conv1d(256, 1024, kernel_size=41, stride=4, padding=20, groups=64),
            nn.Conv1d(1024, 1024, kernel_size=41, stride=4, padding=20, groups=256),
            nn.Conv1d(1024, 1024, kernel_size=5, padding=2),
            nn.Conv1d(1024, 1, kernel_size=3, padding=1),   # logit layer
        ])

    def forward(self, x: torch.Tensor):
        """Returns (logit, list_of_feature_maps)."""
        feats = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.leaky_relu(x, 0.2)
            feats.append(x)
        return x, feats


class MultiScaleDiscriminator(nn.Module):
    """
    Three SubDiscriminators applied at 1x, 2x, and 4x audio downsampling.
    Downsampling is done with a 4-tap average pool (stride 2) applied
    cumulatively: disc0 sees original, disc1 sees pool(x), disc2 sees pool(pool(x)).
    """

    def __init__(self):
        super().__init__()
        self.discs = nn.ModuleList([SubDiscriminator() for _ in range(3)])
        self.pool = nn.AvgPool1d(kernel_size=4, stride=2, padding=2)

    def forward(self, x: torch.Tensor):
        """
        x: [B, 1, T]
        Returns:
            outs:      list of 3 logit tensors
            feats_all: list of 3 feature-map lists (for feature-matching loss)
        """
        outs, feats_all = [], []
        for i, disc in enumerate(self.discs):
            xi = x
            for _ in range(i):
                xi = self.pool(xi)
            out, feats = disc(xi)
            outs.append(out)
            feats_all.append(feats)
        return outs, feats_all


# ─────────────────────────────────────────────────────────────────
# Multi-Resolution STFT Loss (optional extra)
# ─────────────────────────────────────────────────────────────────

class MultiResolutionSTFTLoss(nn.Module):
    """
    Spectral convergence + log-STFT magnitude loss at multiple resolutions.
    Encourages the generator to match the real waveform's spectral structure.
    """

    FFT_SIZES  = (512, 1024, 2048)
    HOP_SIZES  = (120,  256,  512)
    WIN_SIZES  = (240,  600, 1200)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """pred, target: [B, 1, T]"""
        pred = pred.squeeze(1)
        target = target.squeeze(1)
        loss = torch.tensor(0.0, device=pred.device)
        for n_fft, hop, win in zip(self.FFT_SIZES, self.HOP_SIZES, self.WIN_SIZES):
            window = torch.hann_window(win, device=pred.device)
            p_spec = torch.stft(pred,   n_fft, hop, win, window, return_complex=True).abs()
            t_spec = torch.stft(target, n_fft, hop, win, window, return_complex=True).abs()
            # Spectral convergence
            loss += torch.norm(t_spec - p_spec, p='fro') / (torch.norm(t_spec, p='fro') + 1e-9)
            # Log STFT magnitude
            loss += F.l1_loss(p_spec.log().clamp(-100), t_spec.log().clamp(-100))
        return loss / len(self.FFT_SIZES)
