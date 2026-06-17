from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import librosa

BASE = Path('/home/user/work/tf-mp/kaggle/birdclef-2026')
WORK = Path('/home/user/work/tf-mp/kaggle/working_949')
BN_ROOT = Path('/home/user/work/tf-mp/kaggle/birdnet-analyzer-tflite-birdnet_global_6k_v2.4_model_fp32-1-v3')

SR = 32_000
WINDOW_SEC = 5
N_WINDOWS = 12
WINDOW_SAMPLES = SR * WINDOW_SEC

BIRDNET_SR = 48_000
BIRDNET_CHUNK_SEC = 3
BIRDNET_CHUNK_SAMPLES = BIRDNET_SR * BIRDNET_CHUNK_SEC
N_BN_CHUNKS = 20


def find_birdnet_files():
    model_paths = list(BN_ROOT.rglob('*birdnet*fp32*.tflite')) + list(BN_ROOT.rglob('*BirdNET*Model*.tflite'))
    label_paths = list(BN_ROOT.rglob('*Labels.txt')) + list(BN_ROOT.rglob('*labels*.txt'))
    assert model_paths, f'No tflite under {BN_ROOT}'
    assert label_paths, f'No labels under {BN_ROOT}'
    return model_paths[0], label_paths[0]


def main():
    classes = pd.read_csv(BASE / 'sample_submission.csv', nrows=1).columns[1:].tolist()
    N_CLASSES = len(classes)
    assert N_CLASSES == 234, f'Expected 234 classes, got {N_CLASSES}'
    label_to_idx = {c: i for i, c in enumerate(classes)}
    taxonomy = pd.read_csv(BASE / 'taxonomy.csv')

    model_path, labels_path = find_birdnet_files()
    print(f'BirdNET model:  {model_path}')
    print(f'BirdNET labels: {labels_path}')

    from tensorflow.lite.python.interpreter import Interpreter
    interp = Interpreter(model_path=str(model_path), num_threads=4)
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()
    logit_idx = out_det[-1]['index']
    print(f'  input shape:  {in_det["shape"]}')
    print(f'  output shape: {out_det[-1]["shape"]}')

    bn_labels = [l.strip() for l in labels_path.read_text().splitlines() if l.strip()]
    print(f'  labels: {len(bn_labels)} species')

    bn_sci = [lbl.split('_', 1)[0].strip() for lbl in bn_labels]

    tax_sci_to_label = taxonomy.set_index('scientific_name')['primary_label'].to_dict()
    BN_TO_COMP = {}
    for bn_i, sci in enumerate(bn_sci):
        if sci in tax_sci_to_label and tax_sci_to_label[sci] in label_to_idx:
            BN_TO_COMP[bn_i] = label_to_idx[tax_sci_to_label[sci]]

    mapped = set(BN_TO_COMP.values())
    BN_PROXY = {}
    for ci, primary in enumerate(classes):
        if ci in mapped:
            continue
        row = taxonomy[taxonomy['primary_label'] == primary]
        if row.empty:
            continue
        genus = str(row.iloc[0]['scientific_name']).split()[0]
        idxs = [i for i, s in enumerate(bn_sci) if s.startswith(genus + ' ')]
        if idxs:
            BN_PROXY[ci] = idxs

    n_mapped = len(mapped | set(BN_PROXY.keys()))
    print(f'BirdNET -> competition: {len(BN_TO_COMP)} direct + {len(BN_PROXY)} genus-proxy '
          f'= {n_mapped}/{N_CLASSES} classes')

    win_to_chunks = []
    for w in range(N_WINDOWS):
        ws, we = w * 5, (w + 1) * 5
        win_to_chunks.append([j for j in range(N_BN_CHUNKS) if 3*j < we and 3*(j+1) > ws])

    pseudo_files_txt = Path(os.environ.get('PSEUDO_LIST', str(WORK / 'pseudo_gen' / 'pseudo_test_files.txt')))
    assert pseudo_files_txt.exists(), f'Missing: {pseudo_files_txt}'
    file_list = [l.strip() for l in pseudo_files_txt.read_text().splitlines() if l.strip()]
    print(f'\nProcessing {len(file_list)} soundscape files')

    missing = [f for f in file_list[:10] if not (BASE / 'train_soundscapes' / f).exists()]
    assert not missing, f'Missing soundscape files (first 10): {missing}'

    n_total = len(file_list) * N_WINDOWS
    row_ids = []
    scores = np.zeros((n_total, N_CLASSES), dtype=np.float32)

    t0 = time.time()
    for fi, fn in enumerate(file_list, 1):
        path = BASE / 'train_soundscapes' / fn
        y, sr0 = sf.read(str(path), dtype='float32', always_2d=False)
        if y.ndim == 2:
            y = y.mean(axis=1)
        if sr0 != BIRDNET_SR:
            y = librosa.resample(y, orig_sr=sr0, target_sr=BIRDNET_SR)
        need = 60 * BIRDNET_SR
        y = np.pad(y, (0, need - len(y))) if len(y) < need else y[:need]

        chunks = y.reshape(N_BN_CHUNKS, BIRDNET_CHUNK_SAMPLES)
        chunk_probs = np.zeros((N_BN_CHUNKS, len(bn_labels)), dtype=np.float32)
        for j in range(N_BN_CHUNKS):
            interp.set_tensor(in_det['index'], chunks[j][None, :].astype(np.float32))
            interp.invoke()
            logits = interp.get_tensor(logit_idx)[0]
            chunk_probs[j] = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))

        stem = path.stem
        base = (fi - 1) * N_WINDOWS
        for w, clist in enumerate(win_to_chunks):
            wp = chunk_probs[clist].max(axis=0)
            r = base + w
            row_ids.append(f'{stem}_{(w + 1) * 5}')
            for bn_i, ci in BN_TO_COMP.items():
                if wp[bn_i] > scores[r, ci]:
                    scores[r, ci] = wp[bn_i]
            for ci, bn_idxs in BN_PROXY.items():
                v = wp[bn_idxs].max()
                if v > scores[r, ci]:
                    scores[r, ci] = v

        if fi == 1 or fi % 50 == 0 or fi == len(file_list):
            elapsed = time.time() - t0
            eta = elapsed * (len(file_list) - fi) / fi
            print(f'  [{fi}/{len(file_list)}] {elapsed:.0f}s elapsed, eta {eta:.0f}s ({eta/60:.1f}m)')

    elapsed = time.time() - t0
    print(f'\nBirdNET inference done in {elapsed:.0f}s ({elapsed/60:.1f}m)')

    out_df = pd.DataFrame(scores, columns=classes)
    out_df.insert(0, 'row_id', row_ids)
    out_path = Path(os.environ.get('PSEUDO_OUT', str(WORK / 'pseudo_gen' / 'submission_birdnet.csv')))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f'Saved {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)')

    coverage = (scores > 0.01).any(axis=0).sum()
    high_conf = (scores > 0.5).any(axis=0).sum()
    meta = {
        'n_files': len(file_list),
        'n_rows': len(row_ids),
        'n_classes': N_CLASSES,
        'classes_covered_at_>0.01': int(coverage),
        'classes_with_any_>0.5_prediction': int(high_conf),
        'classes_mapped_directly': len(BN_TO_COMP),
        'classes_with_genus_proxy': len(BN_PROXY),
        'wall_time_seconds': float(elapsed),
        'birdnet_model': str(model_path),
        'birdnet_labels': str(labels_path),
    }
    (WORK / 'e13_birdnet_pseudo_result.json').write_text(json.dumps(meta, indent=2))
    print(f'Coverage (any class with >0.01 anywhere): {coverage}/{N_CLASSES}')
    print(f'High-conf coverage (>0.5 anywhere):       {high_conf}/{N_CLASSES}')


if __name__ == '__main__':
    main()
