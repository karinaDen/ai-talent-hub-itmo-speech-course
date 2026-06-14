"""
Assignment 2 — Evaluation script
Runs Tasks 1–7b and saves plots + results to ./results/

Usage:
    python evaluate.py [--data-root .] [--lm lm/3-gram.pruned.1e-7.arpa]
                       [--beam-width 10] [--max-samples 200]

Set --max-samples to a small number (e.g. 10) for a quick smoke test.
"""

import argparse
import csv
import os
import time

import kenlm
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
# Acoustic model — loaded ONCE for the whole script
# ─────────────────────────────────────────────
_decoder: Wav2Vec2Decoder = None   # populated in main()

# LM cache: path -> kenlm.Model  (each .arpa loaded only once)
_lm_cache: dict = {None: None}

def _get_lm(path):
    if path not in _lm_cache:
        print(f"  [LM] loading {path} ...")
        _lm_cache[path] = kenlm.Model(path)
    return _lm_cache[path]


def _cfg(lm_path=None, beam_width=10, alpha=1.0, beta=1.0, temperature=1.0):
    """Reconfigure the shared decoder in-place (no model reload)."""
    _decoder.lm_model = _get_lm(lm_path)
    _decoder.beam_width = beam_width
    _decoder.alpha = alpha
    _decoder.beta = beta
    _decoder.temperature = temperature
    return _decoder


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
# Evaluation
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
    return jiwer.wer(refs, hyps), jiwer.cer(refs, hyps), elapsed, hyps


# ─────────────────────────────────────────────
# Plotting
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
    global _decoder

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='.')
    parser.add_argument('--lm', default='lm/3-gram.pruned.1e-7.arpa')
    parser.add_argument('--beam-width', type=int, default=10)
    parser.add_argument('--max-samples', type=int, default=200)
    parser.add_argument('--lm4', default=None,
                        help='Path to 4-gram ARPA LM for Task 5 '
                             '(e.g. lm/4-gram.arpa). Download from openslr.org/11')
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Load acoustic model ONCE ─────────────────────────────────────────────
    print("Loading acoustic model (once for the entire script)...")
    _decoder = Wav2Vec2Decoder(lm_model_path=None)
    print("Model ready.\n")

    # ── Pre-load the 3-gram LM ───────────────────────────────────────────────
    _get_lm(args.lm)

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
    print(f"Earnings22 test:        {len(e22_samples)} samples\n")

    BW = args.beam_width
    LM = args.lm

    # ── Task 1: Greedy ───────────────────────────────────────────────────────
    print("[Task 1] Greedy — LibriSpeech")
    wer, cer, t, _ = evaluate_dataset(_cfg(None, BW), ls_samples, 'greedy')
    print(f"  WER={wer:.4f}  CER={cer:.4f}  ({t:.1f}s)  ref: WER~10.4% CER~3.5%\n")

    # ── Task 2: Beam search — vary beam_width ────────────────────────────────
    print("[Task 2] Beam search — vary beam_width, LibriSpeech")
    bw_results = {}
    for bw in [1, 3, 10, 50]:
        print(f"  beam_width={bw} ...", end=' ', flush=True)
        wer_bw, cer_bw, t_bw, _ = evaluate_dataset(_cfg(None, bw), ls_samples, 'beam')
        bw_results[bw] = (wer_bw, cer_bw, t_bw)
        print(f"WER={wer_bw:.4f}  CER={cer_bw:.4f}  ({t_bw:.1f}s)")

    save_line(
        list(bw_results.keys()),
        {'WER': [v[0] for v in bw_results.values()],
         'CER': [v[1] for v in bw_results.values()]},
        'beam_width', 'Error Rate', 'Beam Width vs Error Rate (LibriSpeech)',
        'task2_beamwidth.png'
    )
    print(f"  ref: WER~9.9% CER~3.4%\n")

    # ── Task 3: Temperature sweep — greedy, LibriSpeech ─────────────────────
    print("[Task 3] Temperature sweep — greedy, LibriSpeech")
    temps = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    temp_wers = []
    for T in temps:
        print(f"  T={T} ...", end=' ', flush=True)
        wer_t, _, _, _ = evaluate_dataset(_cfg(None, BW, temperature=T), ls_samples, 'greedy')
        temp_wers.append(wer_t)
        print(f"WER={wer_t:.4f}")

    save_line(temps, {'WER (greedy)': temp_wers},
              'Temperature', 'WER',
              'Temperature vs WER — Greedy (LibriSpeech)', 'task3_temperature_ls.png')
    print()

    # ── Task 4: Beam + 3-gram LM — alpha/beta grid, LibriSpeech ─────────────
    print("[Task 4] Beam + 3-gram LM — alpha/beta sweep, LibriSpeech")
    alphas = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    betas  = [0.0, 0.5, 1.0, 1.5]
    lm_grid = {}
    for alpha in alphas:
        for beta in betas:
            print(f"  alpha={alpha:<5} beta={beta} ...", end=' ', flush=True)
            wer_ab, _, _, _ = evaluate_dataset(
                _cfg(LM, BW, alpha=alpha, beta=beta), ls_samples, 'beam_lm'
            )
            lm_grid[(alpha, beta)] = wer_ab
            print(f"WER={wer_ab:.4f}")

    df_lm = pd.DataFrame(
        [[lm_grid[(a, b)] for b in betas] for a in alphas],
        index=[str(a) for a in alphas],
        columns=[str(b) for b in betas]
    )
    save_heatmap(df_lm, 'Task 4: WER — beam+3gram LM (LibriSpeech)', 'task4_lm_heatmap.png')
    best_ab = min(lm_grid, key=lm_grid.get)
    best_alpha, best_beta = best_ab
    print(f"  Best: alpha={best_alpha}  beta={best_beta}  WER={lm_grid[best_ab]:.4f}  ref: WER~9.7%\n")

    # ── Task 5: 4-gram LM — compare with 3-gram ──────────────────────────────
    lm4_best_wer = None
    if args.lm4:
        print("[Task 5] 4-gram LM — beam search with best alpha from Task 4, LibriSpeech")
        _get_lm(args.lm4)
        t5_results = {}
        for beta in betas:
            print(f"  4-gram  alpha={best_alpha:<5} beta={beta} ...", end=' ', flush=True)
            wer_t5, cer_t5, _, _ = evaluate_dataset(
                _cfg(args.lm4, BW, alpha=best_alpha, beta=beta), ls_samples, 'beam_lm'
            )
            t5_results[beta] = (wer_t5, cer_t5)
            print(f"WER={wer_t5:.4f}  CER={cer_t5:.4f}")
        best_t5_beta = min(t5_results, key=lambda b: t5_results[b][0])
        lm4_best_wer = t5_results[best_t5_beta][0]
        lm4_best_cer = t5_results[best_t5_beta][1]
        print(f"\n  3-gram best: WER={lm_grid[best_ab]:.4f}  (a={best_alpha}, b={best_beta})")
        print(f"  4-gram best: WER={lm4_best_wer:.4f}  CER={lm4_best_cer:.4f}  (a={best_alpha}, b={best_t5_beta})\n")
    else:
        print("[Task 5] Skipped — pass --lm4 /path/to/4-gram.arpa to enable\n")
        print("  Download: https://openslr.org/resources/11/4-gram.arpa.gz\n")
        print("  Decompress (Windows): python -c \""
              "import gzip,shutil; "
              "shutil.copyfileobj(gzip.open('4-gram.arpa.gz','rb'), open('lm/4-gram.arpa','wb'))"
              "\"\n")

    # ── Task 6: LM rescoring — alpha/beta grid, LibriSpeech ──────────────────
    print("[Task 6] LM rescoring — alpha/beta sweep, LibriSpeech")
    rs_grid = {}
    for alpha in alphas:
        for beta in betas:
            print(f"  alpha={alpha:<5} beta={beta} ...", end=' ', flush=True)
            wer_rs, _, _, _ = evaluate_dataset(
                _cfg(LM, BW, alpha=alpha, beta=beta), ls_samples, 'beam_lm_rescore'
            )
            rs_grid[(alpha, beta)] = wer_rs
            print(f"WER={wer_rs:.4f}")

    df_rs = pd.DataFrame(
        [[rs_grid[(a, b)] for b in betas] for a in alphas],
        index=[str(a) for a in alphas],
        columns=[str(b) for b in betas]
    )
    save_heatmap(df_rs, 'Task 6: WER — LM rescoring (LibriSpeech)', 'task6_rescore_heatmap.png')
    best_rs = min(rs_grid, key=rs_grid.get)
    print(f"  Best: alpha={best_rs[0]}  beta={best_rs[1]}  WER={rs_grid[best_rs]:.4f}  ref: WER~9.6%\n")

    # ── Task 6 qualitative: samples where LM changes the hypothesis ───────────
    print("[Task 6 qualitative] Finding samples where SF or RS differs from beam search...")
    qual_examples = []
    for wav_path, ref in ls_samples:
        audio = load_audio(wav_path)
        hyp_beam = _cfg(None, BW).decode(audio, method='beam')
        hyp_sf   = _cfg(LM, BW, alpha=best_alpha, beta=best_beta).decode(audio, method='beam_lm')
        hyp_rs   = _cfg(LM, BW, alpha=best_rs[0], beta=best_rs[1]).decode(audio, method='beam_lm_rescore')
        if hyp_sf != hyp_beam or hyp_rs != hyp_beam:
            def mark(hyp, beam, ref):
                if hyp == beam:    return '(=beam)'
                if hyp == ref:     return '(correct)'
                return '(changed)'
            qual_examples.append({
                'ref': ref, 'beam': hyp_beam,
                'sf': hyp_sf, 'sf_mark': mark(hyp_sf, hyp_beam, ref),
                'rs': hyp_rs, 'rs_mark': mark(hyp_rs, hyp_beam, ref),
            })
        if len(qual_examples) >= 10:
            break

    qual_path = os.path.join(RESULTS_DIR, 'task6_qualitative.txt')
    with open(qual_path, 'w', encoding='utf-8') as qf:
        for i, ex in enumerate(qual_examples, 1):
            block = (f"--- Example {i} ---\n"
                     f"REF : {ex['ref']}\n"
                     f"BEAM: {ex['beam']}\n"
                     f"SF  : {ex['sf']}  {ex['sf_mark']}\n"
                     f"RS  : {ex['rs']}  {ex['rs_mark']}\n\n")
            print(block, end='')
            qf.write(block)
    print(f"  Saved qualitative examples: {qual_path}\n")

    # ── Task 7: Full comparison table on both test sets ───────────────────────
    print("[Task 7] Comparison table — LibriSpeech + Earnings22")
    rows = []
    configs = [
        ('Greedy',                          'greedy',         None, 1.0,        1.0),
        ('Beam search',                     'beam',           None, 1.0,        1.0),
        (f'Beam+LM shallow (a={best_alpha},b={best_beta})',
                                            'beam_lm',        LM,   best_alpha, best_beta),
        (f'Beam+LM rescore (a={best_rs[0]},b={best_rs[1]})',
                                            'beam_lm_rescore', LM,  best_rs[0], best_rs[1]),
    ]
    for method_name, method_str, lm_path, alpha, beta in configs:
        for ds_name, samples in [('LibriSpeech', ls_samples), ('Earnings22', e22_samples)]:
            print(f"  {ds_name} / {method_name} ...", end=' ', flush=True)
            wer_v, cer_v, _, _ = evaluate_dataset(
                _cfg(lm_path, BW, alpha=alpha, beta=beta), samples, method_str
            )
            rows.append({'Dataset': ds_name, 'Method': method_name,
                         'WER': f'{wer_v:.4f}', 'CER': f'{cer_v:.4f}'})
            print(f"WER={wer_v:.4f}  CER={cer_v:.4f}")

    df_table = pd.DataFrame(rows)
    print("\n" + df_table.to_string(index=False))
    df_table.to_csv(os.path.join(RESULTS_DIR, 'task7_table.csv'), index=False)
    print()

    # ── Task 7b: Temperature sweep — Earnings22 ───────────────────────────────
    print("[Task 7b] Temperature sweep — Earnings22")
    temps_7b = [0.5, 1.0, 1.5, 2.0]
    t7b_greedy_wers, t7b_lm_wers = [], []
    for T in temps_7b:
        print(f"  T={T} ...", end=' ', flush=True)
        wg, _, _, _ = evaluate_dataset(_cfg(None, BW, temperature=T), e22_samples, 'greedy')
        wl, _, _, _ = evaluate_dataset(
            _cfg(LM, BW, alpha=best_alpha, beta=best_beta, temperature=T),
            e22_samples, 'beam_lm'
        )
        t7b_greedy_wers.append(wg)
        t7b_lm_wers.append(wl)
        print(f"greedy WER={wg:.4f}  beam_lm WER={wl:.4f}")

    save_line(
        temps_7b,
        {'Greedy': t7b_greedy_wers,
         f'Beam+LM (a={best_alpha},b={best_beta})': t7b_lm_wers},
        'Temperature', 'WER',
        'Task 7b: Temperature vs WER on Earnings22',
        'task7b_temperature_e22.png'
    )

    print(f"\nAll results saved to: {RESULTS_DIR}")


if __name__ == '__main__':
    main()
