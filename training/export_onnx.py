from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

WORK = Path('/home/user/work/tf-mp/kaggle/working_949')

SR = 32_000
N_MELS = 128
IMG_T = 313


class GeMFreq(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p, dtype=torch.float32))
        self.eps = eps

    def forward(self, x):
        return x.clamp(min=self.eps).pow(self.p).mean(dim=-2, keepdim=True).pow(1.0 / self.p)


class AttHead(nn.Module):
    def __init__(self, in_chans, num_class=234, hidden_chans=1024, p=0.5):
        super().__init__()
        self.pooling = GeMFreq()
        self.dense_layers = nn.Sequential(
            nn.Dropout(p / 2),
            nn.Linear(in_chans, hidden_chans),
            nn.ReLU(),
            nn.Dropout(p),
        )
        self.attention = nn.Conv1d(hidden_chans, num_class, kernel_size=1, bias=True)
        self.fix_scale = nn.Conv1d(hidden_chans, num_class, kernel_size=1, bias=True)

    def forward(self, feat):
        feat = self.pooling(feat).squeeze(-2).permute(0, 2, 1)
        feat = self.dense_layers(feat).permute(0, 2, 1)
        time_att = torch.tanh(self.attention(feat))
        feat_v = self.fix_scale(feat)
        att_w = torch.softmax(time_att, dim=-1)
        clipwise_probs = torch.sum(torch.sigmoid(feat_v) * att_w, dim=-1)
        return clipwise_probs


class NormalizeMelSpec(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, X):
        mean = X.mean((-2, -1), keepdim=True)
        std = X.std((-2, -1), keepdim=True)
        Xstd = (X - mean) / (std + self.eps)
        Xmax = torch.amax(Xstd, dim=(-2, -1), keepdim=True)
        Xmin = torch.amin(Xstd, dim=(-2, -1), keepdim=True)
        return (Xstd - Xmin) / (Xmax - Xmin + self.eps)


class BirdSEDExport(nn.Module):
    def __init__(self, backbone_name, num_classes=234):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=False, in_chans=1,
            features_only=True, drop_path_rate=0.0,
        )
        feat_dim = self.backbone.feature_info.channels()[-1]
        self.head = AttHead(feat_dim, num_classes, hidden_chans=1024, p=0.5)
        self.normalize = NormalizeMelSpec()

    def forward(self, mel):
        mel = self.normalize(mel)
        feat = self.backbone(mel)[-1]
        return self.head(feat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt', help='Path to PyTorch checkpoint (.pt)')
    ap.add_argument('--backbone', default='tf_efficientnetv2_s.in21k_ft_in1k')
    ap.add_argument('--out_onnx', default=None,
                    help='Default: replaces .pt with .onnx next to checkpoint')
    ap.add_argument('--batch_size', type=int, default=12,
                    help='Batch size to export (typical inference batch = 12 windows/file)')
    ap.add_argument('--num_classes', type=int, default=234)
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    assert ckpt_path.exists(), f'Checkpoint not found: {ckpt_path}'

    if args.out_onnx is None:
        out_path = ckpt_path.with_suffix('.onnx')
    else:
        out_path = Path(args.out_onnx)

    print(f'Checkpoint: {ckpt_path}')
    print(f'Output:     {out_path}')

    model = BirdSEDExport(args.backbone, num_classes=args.num_classes)
    state = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f'Load: missing={len(missing)}, unexpected={len(unexpected)}')
    if missing:
        print(f'  Missing keys (first 5): {missing[:5]}')
    if unexpected:
        print(f'  Unexpected keys (first 5): {unexpected[:5]}')
    assert not missing,    f'Missing keys: {missing[:5]}'
    assert not unexpected, f'Unexpected keys: {unexpected[:5]}'
    model.eval()

    dummy = torch.randn(args.batch_size, 1, N_MELS, IMG_T, dtype=torch.float32)
    with torch.no_grad():
        ref_out = model(dummy).numpy()
    print(f'PyTorch ref output: shape={ref_out.shape}, range=[{ref_out.min():.5f}, {ref_out.max():.5f}]')

    print(f'Exporting to ONNX...')
    torch.onnx.export(
        model, dummy, str(out_path),
        input_names=['mel'], output_names=['clipwise_probs'],
        dynamic_axes={'mel': {0: 'batch'}, 'clipwise_probs': {0: 'batch'}},
        opset_version=17, do_constant_folding=True,
        export_params=True,
    )
    size_mb = out_path.stat().st_size / 1e6
    print(f'  Wrote {out_path} ({size_mb:.1f} MB)')

    print(f'Verifying ONNX inference vs PyTorch...')
    try:
        import onnxruntime as ort
    except ImportError:
        print('  WARN: onnxruntime not installed; skipping verification')
        return

    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    sess = ort.InferenceSession(str(out_path), sess_options=so, providers=['CPUExecutionProvider'])

    for bs in [1, 6, 12, 24]:
        x = torch.randn(bs, 1, N_MELS, IMG_T, dtype=torch.float32).numpy()
        t0 = time.time()
        onnx_out = sess.run(None, {'mel': x})[0]
        ort_time = time.time() - t0

        with torch.no_grad():
            torch_out = model(torch.from_numpy(x)).numpy()
        t0 = time.time()
        with torch.no_grad():
            _ = model(torch.from_numpy(x)).numpy()
        torch_time = time.time() - t0

        max_diff = np.abs(onnx_out - torch_out).max()
        print(f'  bs={bs:2d}: max_diff={max_diff:.6f}  ONNX {ort_time*1000:.0f}ms vs Torch {torch_time*1000:.0f}ms')
        assert max_diff < 1e-3, f'Numerical mismatch at bs={bs}: {max_diff}'

    print(f'\n✓ ONNX export verified equivalent to PyTorch within 1e-3 numerical tolerance')
    print(f'✓ Output ready for Kaggle: {out_path} ({size_mb:.1f} MB)')


if __name__ == '__main__':
    main()
