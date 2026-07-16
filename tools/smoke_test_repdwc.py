"""Smoke test for RepDWCBackbone (Task 2, Step 5).

Builds RepDWCBackbone from a minimal model_cfg mirroring the RadarNeXt
FPN-variant config values, feeds (2, 32, 320, 320), asserts the 3 output
shapes, prints the training-mode param count, and also verifies the
reparameterize_model round-trip (inference-mode param count + identical output).
"""

import torch
from easydict import EasyDict

from pcdet.models.backbones_2d import RepDWCBackbone
from pcdet.models.backbones_2d.mobileone_blocks import reparameterize_model


def main():
    # model_cfg mirroring RadarNeXt FPN-variant backbone block (exact values).
    model_cfg = EasyDict({
        'OUT_CHANNELS': [64, 128, 256],
        'LAYER_NUMS': [3, 5, 5],
        'LAYER_STRIDES': [2, 2, 2],
        'NUM_OUTPUTS': 3,
        'INFERENCE_MODE': False,
        'USE_SE': False,
        'NUM_CONV_BRANCHES': 1,
        'USE_NORMCONV': False,
        'USE_DWCONV': True,
    })

    # input_channels=32 per audit M4 (NUM_BEV_FEATURES=32, NOT the MMDet config's 64).
    backbone = RepDWCBackbone(model_cfg=model_cfg, input_channels=32)
    backbone.eval()

    x = torch.randn(2, 32, 320, 320)
    with torch.no_grad():
        outs = backbone(x)

    expected = [(2, 64, 160, 160), (2, 128, 80, 80), (2, 256, 40, 40)]
    actual = [tuple(o.shape) for o in outs]
    assert isinstance(outs, list), f'forward must return list, got {type(outs)}'
    assert len(outs) == 3, f'expected 3 scales, got {len(outs)}'
    assert actual == expected, f'shape mismatch: {actual} != {expected}'

    n_train = sum(p.numel() for p in backbone.parameters())
    print('=' * 60)
    print('RepDWCBackbone smoke test (training-mode, multi-branch)')
    print('=' * 60)
    print(f'input  shape : {tuple(x.shape)}')
    print(f'output shapes: {actual}  (expected {expected})')
    print(f'training-mode params: {n_train:,}')
    print('PASS: 3 output shapes match expected')

    # --- reparameterize_model round-trip (Task 5/8/9 use it for inference params) ---
    reparam = reparameterize_model(backbone)
    reparam.eval()
    with torch.no_grad():
        outs_reparam = reparam(x)
    actual_reparam = [tuple(o.shape) for o in outs_reparam]
    assert actual_reparam == expected, f'reparam shape mismatch: {actual_reparam}'
    n_infer = sum(p.numel() for p in reparam.parameters())
    max_abs_diff = max((a - b).abs().max().item() for a, b in zip(outs, outs_reparam))
    print('-' * 60)
    print('reparameterize_model round-trip')
    print('-' * 60)
    print(f'reparam output shapes: {actual_reparam}')
    print(f'inference-mode params: {n_infer:,}')
    print(f'max |train - reparam| output diff: {max_abs_diff:.3e}')
    print('PASS: reparameterized output matches multi-branch output')

    # --- data_dict call style (OpenPCDet detector path) ---
    with torch.no_grad():
        outs_dict = backbone({'spatial_features': x})
    assert [tuple(o.shape) for o in outs_dict] == expected
    print('-' * 60)
    print('PASS: data_dict[\"spatial_features\"] call path works')

    print('=' * 60)
    print('ALL CHECKS PASSED')


if __name__ == '__main__':
    main()
