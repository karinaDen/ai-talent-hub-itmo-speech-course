import argparse
import csv
import os
import time

import jiwer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchaudio

from wav2vec2decoder import Wav2Vec2Decoder

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')


# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────

def load_manifest(manifest_path: str, base_dir: str, max_samples: int):
    samples = []
    with open(manifest_path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            wav = os.path.join(base_dir, row['path'])
            samples.append((wav, row['text']))
            if len(samples) >= max_samples:
                break
    return samples


def load_audio(path: str) -> torch.Tensor:
    waveform, sr = torchaudio.load(path)
    assert sr == 16000, f"Expected 16 kHz, got {sr} at {path}"
    return waveform


# ─────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────

def evaluate_dataset(decoder: Wav2Vec2Decoder, samples, method: str):
    hyps, refs = [], []
    t0 = time.time()
    for i, (wav_path, ref) in enumerate(samples):
        audio = load_audio(wav_path)
        hyp = decoder.decode(audio, method=method)
        hyps.append(hyp)
        refs.append(ref)
        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{len(samples)} done...")
    elapsed = time.time() - t0
    wer = jiwer.wer(refs, hyps)
    cer = jiwer.cer(refs, hyps)
    return wer, cer, elapsed, hyps


def _make_decoder(lm_path, beam_width, alpha, beta, temperature):
    return Wav2Vec2Decoder(
        lm_model_path=lm_path,
        beam_width=beam_width,
        alpha=alpha,
        beta=beta,
        temperature=temperature,
    )


# ─────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────

def save_line(x, ys_dict, xlabel, ylabel, title, filename):
    plt.figure(figsize=(8, 4))
    for label, y in ys_dict.items():
        plt.plot(x, y, 'o-', linewidth=2, markersize=6, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def save_heatmap(df, title, filename):
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(df.values.astype(float), aspect='auto', cmap='RdYlGn_r')
    ax.set_xticks(range(len(df.columns)))
    ax.set_yticks(range(len(df.index)))
    ax.set_xticklabels(df.columns)
    ax.set_yticklabels(df.index)
    ax.set_xlabel('beta')
    ax.set_ylabel('alpha')
    plt.colorbar(im, ax=ax, label='WER')
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            ax.text(j, i, f'{df.values[i,j]:.3f}', ha='center', va='center',
                    color='black', fontsize=8)
    ax.set_title(title)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='.')
    parser.add_argument('--lm', default='lm/3-gram.pruned.1e-7.arpa')
    parser.add_argument('--beam-width', type=int, default=10)
    parser.add_argument('--max-samples', type=int, default=200)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Load manifests ───────────────────────────────────────────────────────
    ls_samples = load_manifest(
        os.path.join(args.data_root, 'data/librispeech_test_other/manifest.csv'),
        args.data_root, args.max_samples
    )
    e22_samples = load_manifest(
        os.path.join(args.data_root, 'data/earnings22_test/manifest.csv'),
        args.data_root, args.max_samples
    )
    print(f"LibriSpeech test-other: {len(ls_samples)} samples")
    print(f"Earnings22 test:        {len(e22_samples)} samples")

    # ── Task 1: Greedy on LibriSpeech ───────────────────────────────────────
    print("\n[Task 1] Greedy decoding — LibriSpeech test-other")
    dec = _make_decoder(None, args.beam_width, 1.0, 1.0, 1.0)
    wer, cer, t, _ = evaluate_dataset(dec, ls_samples, 'greedy')
    print(f"  WER={wer:.4f}  CER={cer:.4f}  time={t:.1f}s")
    print(f"  Reference: WER~10.4%  CER~3.5%")

    # ── Task 2: Beam search — vary beam_width ────────────────────────────────
    print("\n[Task 2] Beam search — vary beam_width on LibriSpeech")
    bw_results = {}
    for bw in [1, 3, 10, 50]:
        print(f"  beam_width={bw} ...")
        dec = _make_decoder(None, bw, 1.0, 1.0, 1.0)
        wer_bw, cer_bw, t_bw, _ = evaluate_dataset(dec, ls_samples, 'beam')
        bw_results[bw] = (wer_bw, cer_bw, t_bw)
        print(f"    WER={wer_bw:.4f}  CER={cer_bw:.4f}  time={t_bw:.1f}s")

    save_line(
        list(bw_results.keys()),
        {'WER': [v[0] for v in bw_results.values()],
         'CER': [v[1] for v in bw_results.values()]},
        'beam_width', 'Error Rate', 'Beam Width vs Error Rate (LibriSpeech)',
        'task2_beamwidth.png'
    )

    # ── Task 3: Temperature sweep — greedy, LibriSpeech ─────────────────────
    print("\n[Task 3] Temperature sweep — greedy, LibriSpeech")
    temps = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    temp_wers = []
    for T in temps:
        print(f"  T={T} ...")
        dec = _make_decoder(None, args.beam_width, 1.0, 1.0, T)
        wer_t, _, _, _ = evaluate_dataset(dec, ls_samples, 'greedy')
        temp_wers.append(wer_t)
        print(f"    WER={wer_t:.4f}")

    save_line(temps, {'WER (greedy)': temp_wers},
              'Temperature', 'WER',
              'Temperature vs WER — Greedy (LibriSpeech)', 'task3_temperature_ls.png')

    # ── Task 4: Beam + 3-gram LM — alpha/beta grid on LibriSpeech ────────────
    print("\n[Task 4] Beam + 3-gram LM shallow fusion — alpha/beta sweep, LibriSpeech")
    alphas = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    betas = [0.0, 0.5, 1.0, 1.5]
    lm_grid = {}
    for alpha in alphas:
        for beta in betas:
            print(f"  alpha={alpha}  beta={beta} ...")
            dec = _make_decoder(args.lm, args.beam_width, alpha, beta, 1.0)
            wer_ab, _, _, _ = evaluate_dataset(dec, ls_samples, 'beam_lm')
            lm_grid[(alpha, beta)] = wer_ab
            print(f"    WER={wer_ab:.4f}")

    df_lm = pd.DataFrame(
        [[lm_grid[(a, b)] for b in betas] for a in alphas],
        index=[str(a) for a in alphas],
        columns=[str(b) for b in betas]
    )
    save_heatmap(df_lm, 'Task 4: WER heatmap — beam+3gram LM (LibriSpeech)', 'task4_lm_heatmap.png')
    best_ab = min(lm_grid, key=lm_grid.get)
    print(f"  Best: alpha={best_ab[0]}  beta={best_ab[1]}  WER={lm_grid[best_ab]:.4f}")
    print(f"  Reference: WER~9.7%  CER~3.4%")

    best_alpha, best_beta = best_ab

    # ── Task 6: LM rescoring — alpha/beta grid on LibriSpeech ────────────────
    print("\n[Task 6] LM rescoring — alpha/beta sweep, LibriSpeech")
    rs_grid = {}
    for alpha in alphas:
        for beta in betas:
            print(f"  alpha={alpha}  beta={beta} ...")
            dec = _make_decoder(args.lm, args.beam_width, alpha, beta, 1.0)
            wer_rs, _, _, _ = evaluate_dataset(dec, ls_samples, 'beam_lm_rescore')
            rs_grid[(alpha, beta)] = wer_rs
            print(f"    WER={wer_rs:.4f}")

    df_rs = pd.DataFrame(
        [[rs_grid[(a, b)] for b in betas] for a in alphas],
        index=[str(a) for a in alphas],
        columns=[str(b) for b in betas]
    )
    save_heatmap(df_rs, 'Task 6: WER heatmap — LM rescoring (LibriSpeech)', 'task6_rescore_heatmap.png')
    best_rs = min(rs_grid, key=rs_grid.get)
    print(f"  Best rescore: alpha={best_rs[0]}  beta={best_rs[1]}  WER={rs_grid[best_rs]:.4f}")
    print(f"  Reference: WER~9.6%  CER~3.3%")

    # ── Task 7: Full comparison table on both test sets ───────────────────────
    print("\n[Task 7] Full comparison table — LibriSpeech + Earnings22")
    rows = []
    for name, samples in [('LibriSpeech', ls_samples), ('Earnings22', e22_samples)]:
        for method_name, method_str, lm_path, alpha, beta in [
            ('Greedy',              'greedy',          None,     1.0,        1.0),
            ('Beam search',         'beam',            None,     1.0,        1.0),
            (f'Beam+3gram (SF a={best_alpha} b={best_beta})', 'beam_lm', args.lm, best_alpha, best_beta),
            (f'Beam+3gram (RS a={best_rs[0]} b={best_rs[1]})', 'beam_lm_rescore', args.lm, best_rs[0], best_rs[1]),
        ]:
            print(f"  {name} / {method_name} ...")
            dec = _make_decoder(lm_path, args.beam_width, alpha, beta, 1.0)
            wer_v, cer_v, _, hyps = evaluate_dataset(dec, samples, method_str)
            rows.append({'Dataset': name, 'Method': method_name,
                         'WER': f'{wer_v:.4f}', 'CER': f'{cer_v:.4f}'})
            print(f"    WER={wer_v:.4f}  CER={cer_v:.4f}")

    df_table = pd.DataFrame(rows)
    print("\n" + df_table.to_string(index=False))
    df_table.to_csv(os.path.join(RESULTS_DIR, 'task7_table.csv'), index=False)

    # ── Task 7b: Temperature sweep — Earnings22 with greedy + beam_lm ────────
    print("\n[Task 7b] Temperature sweep on Earnings22")
    temps_7b = [0.5, 1.0, 1.5, 2.0]
    t7b_greedy_wers, t7b_lm_wers = [], []
    for T in temps_7b:
        print(f"  T={T} ...")
        dec_g = _make_decoder(None, args.beam_width, 1.0, 1.0, T)
        wg, _, _, _ = evaluate_dataset(dec_g, e22_samples, 'greedy')
        dec_lm = _make_decoder(args.lm, args.beam_width, best_alpha, best_beta, T)
        wl, _, _, _ = evaluate_dataset(dec_lm, e22_samples, 'beam_lm')
        t7b_greedy_wers.append(wg)
        t7b_lm_wers.append(wl)
        print(f"    greedy WER={wg:.4f}  beam_lm WER={wl:.4f}")

    save_line(
        temps_7b,
        {'Greedy': t7b_greedy_wers, f'Beam+LM (a={best_alpha},b={best_beta})': t7b_lm_wers},
        'Temperature', 'WER',
        'Task 7b: Temperature vs WER on Earnings22',
        'task7b_temperature_e22.png'
    )

    print(f"\nAll results saved to: {RESULTS_DIR}")


if __name__ == '__main__':
    main()
