"""
GAN training loop for the MelGAN-style neural vocoder.

Usage (Colab / local):
    python train.py \\
        --data-root /content \\
        --out-dir   checkpoints \\
        --epochs    300 \\
        --batch-size 16 \\
        --device    cuda

Checkpoints are saved every --save-every epochs to --out-dir.
Resume training by passing --resume checkpoints/vocoder_epochXXX.pt.

Losses
------
- Discriminator : LSGAN  (real→1, fake→0)
- Generator     : LSGAN adversarial + feature matching + L1 on raw waveform
                  + optional multi-resolution STFT loss (--use-stft-loss)

Reference values after ~100 epochs on T4 GPU (batch 16, ~30 min):
  The audio will sound noisy but word-like.
After ~300 epochs (~1.5 h on T4):
  Speech is intelligible; some metallic artifacts remain.
"""

import argparse
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from dataset import LJSpeechVocoderDataset, HOP_LENGTH, SEGMENT_FRAMES
from vocoder import Generator, MultiScaleDiscriminator, MultiResolutionSTFTLoss


# ── Loss helpers ─────────────────────────────────────────────────────────────

def disc_loss(real_outs, fake_outs):
    """LSGAN discriminator: (D(real)-1)^2 + D(fake)^2."""
    loss = torch.tensor(0.0, device=real_outs[0].device)
    for r, f in zip(real_outs, fake_outs):
        loss += torch.mean((r - 1.0) ** 2) + torch.mean(f ** 2)
    return loss


def gen_adv_loss(fake_outs):
    """LSGAN generator adversarial: (D(fake)-1)^2."""
    loss = torch.tensor(0.0, device=fake_outs[0].device)
    for f in fake_outs:
        loss += torch.mean((f - 1.0) ** 2)
    return loss


def feat_match_loss(real_feats_all, fake_feats_all):
    """L1 on intermediate discriminator features (all layers except logit)."""
    loss = torch.tensor(0.0, device=real_feats_all[0][0].device)
    for real_feats, fake_feats in zip(real_feats_all, fake_feats_all):
        for rf, ff in zip(real_feats[:-1], fake_feats[:-1]):
            loss += F.l1_loss(ff, rf.detach())
    return loss


# ── Training ─────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device(args.device)

    # ── Models ───────────────────────────────────────────────────────────────
    G = Generator(n_mels=80).to(device)
    D = MultiScaleDiscriminator().to(device)

    stft_loss_fn = MultiResolutionSTFTLoss().to(device) if args.use_stft_loss else None

    g_params = sum(p.numel() for p in G.parameters() if p.requires_grad)
    d_params = sum(p.numel() for p in D.parameters() if p.requires_grad)
    print(f"Generator:     {g_params:,} parameters")
    print(f"Discriminator: {d_params:,} parameters")

    # ── Optimizers ───────────────────────────────────────────────────────────
    g_opt = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.9))
    d_opt = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.9))

    # LR schedulers — halve LR every 100 epochs
    g_sched = torch.optim.lr_scheduler.StepLR(g_opt, step_size=100, gamma=0.5)
    d_sched = torch.optim.lr_scheduler.StepLR(d_opt, step_size=100, gamma=0.5)

    start_epoch = 1

    # ── Resume ───────────────────────────────────────────────────────────────
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        G.load_state_dict(ckpt['G'])
        D.load_state_dict(ckpt['D'])
        g_opt.load_state_dict(ckpt['g_opt'])
        d_opt.load_state_dict(ckpt['d_opt'])
        start_epoch = ckpt['epoch'] + 1
        print(f"Resumed from epoch {ckpt['epoch']}")

    # ── Data ─────────────────────────────────────────────────────────────────
    full_ds = LJSpeechVocoderDataset(args.data_root, segment_frames=SEGMENT_FRAMES,
                                     download=True)
    n_val   = min(500, len(full_ds) // 10)
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=(device.type == 'cuda'),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=(device.type == 'cuda'),
        drop_last=True,
    )

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\nTraining for {args.epochs} epochs on {device}  "
          f"(train={n_train}, val={n_val}, batch={args.batch_size})\n")

    for epoch in range(start_epoch, args.epochs + 1):
        G.train(); D.train()
        t0 = time.time()
        g_sum = d_sum = 0.0
        n_batches = 0

        for mel, audio in train_loader:
            mel        = mel.to(device)               # [B, N_MELS, T_mel]
            audio_real = audio.unsqueeze(1).to(device) # [B, 1, T_audio]
            T          = audio_real.shape[-1]

            # Forward G
            audio_fake = G(mel)[..., :T]              # [B, 1, T_audio]  (trim if +1 frame)

            # ── Discriminator step ────────────────────────────────────────
            d_opt.zero_grad()
            real_outs, _          = D(audio_real)
            fake_outs_detach, _   = D(audio_fake.detach())
            loss_d = disc_loss(real_outs, fake_outs_detach)
            loss_d.backward()
            d_opt.step()

            # ── Generator step ────────────────────────────────────────────
            g_opt.zero_grad()
            fake_outs, fake_feats  = D(audio_fake)
            _,          real_feats = D(audio_real)

            loss_adv = gen_adv_loss(fake_outs)
            loss_fm  = feat_match_loss(real_feats, fake_feats)
            loss_l1  = F.l1_loss(audio_fake, audio_real)

            loss_g   = loss_adv + 10.0 * loss_fm + args.lambda_l1 * loss_l1

            if stft_loss_fn is not None:
                loss_g = loss_g + stft_loss_fn(audio_fake, audio_real)

            loss_g.backward()
            g_opt.step()

            g_sum += loss_g.item()
            d_sum += loss_d.item()
            n_batches += 1

        g_sched.step()
        d_sched.step()

        elapsed = time.time() - t0
        print(f"Epoch {epoch:04d}/{args.epochs} | "
              f"G={g_sum/n_batches:.4f}  D={d_sum/n_batches:.4f} | "
              f"{elapsed:.0f}s | lr={g_opt.param_groups[0]['lr']:.2e}")

        # ── Validation (L1 on waveform) ───────────────────────────────────
        if epoch % args.val_every == 0:
            G.eval()
            val_l1 = 0.0
            n_val_batches = 0
            with torch.no_grad():
                for mel, audio in val_loader:
                    mel        = mel.to(device)
                    audio_real = audio.unsqueeze(1).to(device)
                    audio_fake = G(mel)[..., :audio_real.shape[-1]]
                    val_l1    += F.l1_loss(audio_fake, audio_real).item()
                    n_val_batches += 1
            print(f"  → val L1 = {val_l1/n_val_batches:.4f}")

        # ── Checkpoint ───────────────────────────────────────────────────
        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(args.out_dir, f'vocoder_epoch{epoch:04d}.pt')
            torch.save({
                'G':     G.state_dict(),
                'D':     D.state_dict(),
                'g_opt': g_opt.state_dict(),
                'd_opt': d_opt.state_dict(),
                'epoch': epoch,
            }, ckpt_path)
            print(f"  Saved: {ckpt_path}")

    print(f"\nTraining complete. Checkpoints in: {args.out_dir}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Train MelGAN neural vocoder on LJSpeech'
    )
    parser.add_argument('--data-root',    default='/content',
                        help='Root directory for LJSpeech download')
    parser.add_argument('--out-dir',      default='checkpoints')
    parser.add_argument('--resume',       default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--epochs',       type=int, default=300)
    parser.add_argument('--batch-size',   type=int, default=16)
    parser.add_argument('--lr',           type=float, default=1e-4)
    parser.add_argument('--lambda-l1',    type=float, default=10.0,
                        help='Weight for time-domain L1 loss')
    parser.add_argument('--use-stft-loss', action='store_true',
                        help='Also add multi-resolution STFT loss')
    parser.add_argument('--workers',      type=int, default=2)
    parser.add_argument('--save-every',   type=int, default=50,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--val-every',    type=int, default=10,
                        help='Run validation every N epochs')
    parser.add_argument('--device',       default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
