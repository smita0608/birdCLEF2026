from __future__ import annotations

import json
from pathlib import Path

SRC = Path('/home/user/work/tf-mp/kaggle/submissions/auc0.949birdclef-2026-v6.ipynb')
OUT = Path('/home/user/work/tf-mp/kaggle/submissions/auc0.949_TTA.ipynb')

OLD_SED_BLOCK = '''        def file_to_sed_chunks(path):
            y, sr0 = sf.read(str(path), dtype="float32", always_2d=False)
            if y.ndim == 2: y = y.mean(axis=1)
            if sr0 != SR: y = librosa.resample(y, orig_sr=sr0, target_sr=SR)
            n = 60 * SR
            if len(y) < n: y = np.pad(y, (0, n - len(y)))
            else:          y = y[:n]
            chunks = y.reshape(N_WINDOWS, WINDOW_SAMPLES)
            ends   = np.arange(1, N_WINDOWS + 1) * WINDOW_SEC
            return chunks, ends'''

NEW_SED_BLOCK = '''        # ── TTA SHIFTS (E8 modification) ─────────────────────────────
        # Run SED at 3 temporal shifts: {0, +1s, -1s}. Averaging across
        # shifts gives windows extra chances to catch vocalizations near
        # the window boundary. Documented +0.005-0.012 AUC in 2025 top-2%.
        SED_TTA_SHIFTS = [0, 1, -1]   # set to [0] to disable TTA
        # ─────────────────────────────────────────────────────────────────
        def file_to_sed_chunks(path):
            y, sr0 = sf.read(str(path), dtype="float32", always_2d=False)
            if y.ndim == 2: y = y.mean(axis=1)
            if sr0 != SR: y = librosa.resample(y, orig_sr=sr0, target_sr=SR)
            n = 60 * SR
            if len(y) < n: y = np.pad(y, (0, n - len(y)))
            else:          y = y[:n]
            chunks = y.reshape(N_WINDOWS, WINDOW_SAMPLES)
            ends   = np.arange(1, N_WINDOWS + 1) * WINDOW_SEC
            return chunks, ends

        def file_to_sed_chunks_tta(path):
            """Return [chunks_for_shift_0, chunks_for_shift_+1s, ...] aligned to
            the same `ends` as the non-TTA path. Boundary samples padded with zeros."""
            y, sr0 = sf.read(str(path), dtype="float32", always_2d=False)
            if y.ndim == 2: y = y.mean(axis=1)
            if sr0 != SR: y = librosa.resample(y, orig_sr=sr0, target_sr=SR)
            n = 60 * SR
            if len(y) < n: y = np.pad(y, (0, n - len(y)))
            else:          y = y[:n]
            max_shift = max(abs(s) for s in SED_TTA_SHIFTS) * SR
            y_padded = np.pad(y, (max_shift, max_shift), mode="constant")
            chunk_sets = []
            for shift_sec in SED_TTA_SHIFTS:
                shift_samples = int(shift_sec * SR)
                start = max_shift + shift_samples
                y_shifted = y_padded[start:start + n]
                chunks = y_shifted.reshape(N_WINDOWS, WINDOW_SAMPLES)
                chunk_sets.append(chunks)
            ends = np.arange(1, N_WINDOWS + 1) * WINDOW_SEC
            return chunk_sets, ends'''

OLD_SED_LOOP = '        for i, path in enumerate(test_paths, 1):\n' \
               '            chunks, ends = file_to_sed_chunks(path)\n' \
               '            mel = audio_to_mel(chunks)\n' \
               '            p_sum = np.zeros((len(chunks), N_CLASSES), dtype=np.float32)\n' \
               '        \n' \
               '            for sess in sed_sessions:\n' \
               '                outs = sess.run(None, {sess.get_inputs()[0].name: mel})\n' \
               '                clip_logits = outs[0]             # (12, 234)\n' \
               '                frame_max   = outs[1].max(axis=1) # (12, 234)\n' \
               '                p_sum += 0.5 * sigmoid_sed(clip_logits) + 0.5 * sigmoid_sed(frame_max)\n' \
               '        \n' \
               '            p_mean = p_sum / len(sed_sessions)'

NEW_SED_LOOP = '        for i, path in enumerate(test_paths, 1):\n' \
               '            # E8 TTA: iterate over shifts, average predictions\n' \
               '            chunk_sets, ends = file_to_sed_chunks_tta(path)\n' \
               '            p_sum = np.zeros((N_WINDOWS, N_CLASSES), dtype=np.float32)\n' \
               '            n_subpreds = 0\n' \
               '        \n' \
               '            for chunks in chunk_sets:\n' \
               '                mel = audio_to_mel(chunks)\n' \
               '                for sess in sed_sessions:\n' \
               '                    outs = sess.run(None, {sess.get_inputs()[0].name: mel})\n' \
               '                    clip_logits = outs[0]             # (12, 234)\n' \
               '                    frame_max   = outs[1].max(axis=1) # (12, 234)\n' \
               '                    p_sum += 0.5 * sigmoid_sed(clip_logits) + 0.5 * sigmoid_sed(frame_max)\n' \
               '                    n_subpreds += 1\n' \
               '        \n' \
               '            p_mean = p_sum / max(1, n_subpreds)'


def main():
    with open(SRC) as f:
        nb = json.load(f)
    src = ''.join(nb['cells'][12]['source'])

    assert OLD_SED_BLOCK in src, 'OLD_SED_BLOCK not found — notebook may have been edited'
    src_v2 = src.replace(OLD_SED_BLOCK, NEW_SED_BLOCK, 1)
    assert OLD_SED_LOOP in src_v2, 'OLD_SED_LOOP not found'
    src_v3 = src_v2.replace(OLD_SED_LOOP, NEW_SED_LOOP, 1)
    assert 'SED_TTA_SHIFTS' in src_v3, 'TTA tag missing'

    lines = src_v3.splitlines(keepends=True)
    nb['cells'][12]['source'] = lines

    title = ''.join(nb['cells'][0]['source']) if nb['cells'][0]['cell_type'] == 'markdown' else ''
    new_title = ('# BirdCLEF 2026: auc0.949 + SED TTA (3 shifts ±1s)\n\n'
                 'Modification of auc0.949birdclef-2026-v6.ipynb: SED inference now runs '
                 'at 3 temporal shifts (0, +1s, -1s) per file and averages predictions. '
                 'Documented in 2025 top-2% writeup as +0.005-0.012 AUC. '
                 'Estimated runtime impact: +20-25 min on SED step.\n')
    nb['cells'][0]['source'] = new_title.splitlines(keepends=True)

    for c in nb['cells']:
        if c['cell_type'] == 'code':
            c['outputs'] = []
            c['execution_count'] = None

    OUT.write_text(json.dumps(nb, indent=1))
    print(f'Wrote {OUT}: {OUT.stat().st_size / 1024:.0f} KB')

    import re
    new_src = json.loads(OUT.read_text())['cells'][12]['source']
    flat = ''.join(new_src)
    assert 'SED_TTA_SHIFTS = [0, 1, -1]' in flat, 'TTA config missing'
    assert 'file_to_sed_chunks_tta' in flat, 'TTA function missing'
    assert 'for chunks in chunk_sets:' in flat, 'TTA loop missing'
    print('All TTA insertion points verified.')


if __name__ == '__main__':
    main()
