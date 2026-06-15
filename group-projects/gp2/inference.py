"""
Inference script: text -> mel (via FastPitch) -> wav (via trained vocoder).

Usage:
    python inference.py \\
        --checkpoint checkpoints/vocoder_epoch0300.pt \\
        --sentences  test_sentences.txt \\
        --out-dir    samples/

Generates one .wav file per sentence in --sentences.
"""

import argparse
import os
import sys

import numpy as np
import torch
import torchaudio

# ── Import project modules ───────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vocoder import Generator
from dataset import N_MELS, LOG_EPS


def load_generator(checkpoint_path: str, device: str) -> Generator:
    G = Generator(n_mels=N_MELS).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    G.load_state_dict(ckpt['G'])
    G.eval()
    return G


def load_tts():
    """Load the FastPitch TextToSpecConverter (lazy import to avoid import errors
    when TTS is not installed)."""
    from t2spec_converter import TextToSpecConverter
    return TextToSpecConverter()


@torch.no_grad()
def text_to_wav(text: str, t2s, G: Generator, device: str) -> torch.Tensor:
    """
    Convert text to waveform.

    1. FastPitch converts text -> log mel  [80, T]  (numpy)
    2. Generator converts  log mel -> wav  [1, T*256]

    t2spec_converter.text2spec() already returns the mel in log scale
    (Coqui-TTS AudioProcessor with log_func='np.log', signal_norm=False),
    which matches the log mel scale used during vocoder training.
    """
    # text -> log mel spectrogram
    mel_np  = t2s.text2spec(text)                          # [80, T]  numpy float32
    mel     = torch.from_numpy(mel_np).unsqueeze(0).to(device)  # [1, 80, T]

    # vocoder: log mel -> waveform
    wav = G(mel)   # [1, 1, T_wav]
    return wav.squeeze(0).cpu()   # [1, T_wav]


def main():
    parser = argparse.ArgumentParser(
        description='Generate audio from text using trained MelGAN vocoder'
    )
    parser.add_argument('--checkpoint', required=True,
                        help='Path to vocoder checkpoint (.pt)')
    parser.add_argument('--sentences', default='test_sentences.txt',
                        help='Text file with one sentence per line')
    parser.add_argument('--out-dir', default='samples',
                        help='Output directory for generated .wav files')
    parser.add_argument('--device',
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device

    # Load models
    print("Loading TTS (FastPitch text -> mel)...")
    t2s = load_tts()

    print(f"Loading vocoder from {args.checkpoint}...")
    G = load_generator(args.checkpoint, device)

    print(f"Device: {device}\n")

    # Load sentences
    with open(args.sentences, encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]

    # Generate
    for i, text in enumerate(sentences, 1):
        print(f"[{i}/{len(sentences)}]  {text}")
        wav = text_to_wav(text, t2s, G, device)       # [1, T]
        out_path = os.path.join(args.out_dir, f'sample_{i:02d}.wav')
        torchaudio.save(out_path, wav, 22050)
        duration = wav.shape[-1] / 22050
        print(f"           -> {out_path}  ({duration:.2f}s)")

    print(f"\nDone. {len(sentences)} samples saved to '{args.out_dir}/'")


if __name__ == '__main__':
    main()
