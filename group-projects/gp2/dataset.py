"""
LJSpeech dataset for vocoder training.

Mel-spectrogram parameters are taken from the Coqui-TTS FastPitch config:
  sample_rate=22050, n_fft=1024, hop_length=256, n_mels=80,
  f_min=0.0, f_max=8000.0, power=1.5, log_func='np.log'

Each __getitem__ returns:
  log_mel : [N_MELS, SEGMENT_FRAMES]  float32 log mel spectrogram
  audio   : [SEGMENT_FRAMES * HOP_LENGTH]  float32 waveform in [-1, 1]
"""

import os
import random

import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset

# ── Audio / mel config (must match t2spec_converter TTS config) ─────────────
SAMPLE_RATE     = 22050
N_FFT           = 1024
WIN_LENGTH      = 1024
HOP_LENGTH      = 256
N_MELS          = 80
F_MIN           = 0.0
F_MAX           = 8000.0
MEL_POWER       = 1.5      # Coqui-TTS: |STFT|^1.5 before mel filterbank
LOG_EPS         = 1e-5     # Coqui-TTS: log(max(1e-5, mel))

# ── Training segment size ───────────────────────────────────────────────────
SEGMENT_FRAMES  = 32       # mel frames per training item
SEGMENT_SAMPLES = SEGMENT_FRAMES * HOP_LENGTH  # = 8 192 audio samples


# ── Mel filterbank (computed once, shared across workers) ───────────────────
_MEL_FB: dict = {}   # keyed by device string


def _get_mel_fb(device: torch.device) -> torch.Tensor:
    key = str(device)
    if key not in _MEL_FB:
        _MEL_FB[key] = torchaudio.functional.melscale_fbanks(
            n_freqs=N_FFT // 2 + 1,
            f_min=F_MIN,
            f_max=F_MAX,
            n_mels=N_MELS,
            sample_rate=SAMPLE_RATE,
            norm=None,
            mel_scale='htk',      # matches librosa default used by Coqui-TTS
        ).to(device)
    return _MEL_FB[key]


def compute_log_mel(wav: torch.Tensor) -> torch.Tensor:
    """
    Compute log mel spectrogram matching Coqui-TTS AudioProcessor.

    wav: [T]  float32 mono waveform
    Returns: [N_MELS, T_mel]  log mel (natural log)

    Formula:  log(max(1e-5,  mel_fb @ |STFT(wav)|^1.5))
    """
    window = torch.hann_window(WIN_LENGTH, device=wav.device)
    spec = torch.stft(
        wav, N_FFT, HOP_LENGTH, WIN_LENGTH, window,
        center=True, return_complex=True,
    )                                    # [n_freqs, T_mel]
    mag  = spec.abs().pow(MEL_POWER)     # [n_freqs, T_mel]

    mel_fb = _get_mel_fb(wav.device)     # [n_freqs, N_MELS]
    mel    = torch.matmul(mag.T, mel_fb).T   # [N_MELS, T_mel]
    return torch.log(mel.clamp(min=LOG_EPS))


# ── Dataset ─────────────────────────────────────────────────────────────────

class LJSpeechVocoderDataset(Dataset):
    """
    Streams LJSpeech from disk; returns fixed-length (mel, audio) pairs.

    File durations are read from WAV headers via soundfile (no audio loaded
    at init time), so startup is fast.
    """

    def __init__(self, data_root: str, segment_frames: int = SEGMENT_FRAMES,
                 download: bool = True, wav_dir: str = None):
        self.segment_frames  = segment_frames
        self.segment_samples = segment_frames * HOP_LENGTH

        if wav_dir is None:
            wav_dir = os.path.join(data_root, 'LJSpeech-1.1', 'wavs')
            if not os.path.isdir(wav_dir):
                torchaudio.datasets.LJSPEECH(data_root, download=download)
        min_samp = self.segment_samples + WIN_LENGTH   # safety margin

        self.wav_paths = sorted([
            os.path.join(wav_dir, fn)
            for fn in os.listdir(wav_dir)
            if fn.endswith('.wav')
            and sf.info(os.path.join(wav_dir, fn)).frames >= min_samp
        ])
        print(f"LJSpeech vocoder dataset: {len(self.wav_paths)} usable files "
              f"(segment={segment_frames} mel frames = {self.segment_samples} samples)")

    def __len__(self) -> int:
        return len(self.wav_paths)

    def __getitem__(self, idx: int):
        # Load audio from disk
        data, sr = sf.read(self.wav_paths[idx], dtype='float32', always_2d=False)
        wav = torch.from_numpy(data)
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)

        # Random crop
        max_start = wav.shape[0] - self.segment_samples
        start     = random.randint(0, max(0, max_start))
        audio     = wav[start : start + self.segment_samples]   # [segment_samples]

        # Pad if the file was shorter than segment (rare edge case)
        if audio.shape[0] < self.segment_samples:
            audio = torch.nn.functional.pad(
                audio, (0, self.segment_samples - audio.shape[0])
            )

        # Compute log mel; with center=True STFT may produce segment_frames+1 frames
        # → trim to exactly segment_frames so generator output = audio length
        log_mel = compute_log_mel(audio)[:, : self.segment_frames]  # [N_MELS, frames]

        return log_mel, audio
