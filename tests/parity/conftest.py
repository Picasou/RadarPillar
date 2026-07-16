"""Shared parity-test infrastructure.

Provides:
  * fixed-seed synthetic input generators,
  * a weight-alignment helper that copies one module's state_dict into
    another by exact key match (so both sides hold IDENTICAL weights —
    required by Task 4.5 premise #1),
  * an ``allclose`` assertor with first-mismatch drilldown (prints the
    index, the two values, and the relative error).

All defaults are CPU-friendly. Tests that need CUDA call ``.cuda()`` on
the modules they construct; the helpers stay device-agnostic.
"""

import torch

# Default RNG seed mandated by execution protocol §5 (input alignment).
SEED = 0

# Default parity tolerance (Task 4.5 brief: fp32 atol=1e-4 rtol=1e-3).
DEFAULT_ATOL = 1e-4
DEFAULT_RTOL = 1e-3
# Looser tolerance for modules with grid_sample / deconv arithmetic.
LOOSE_ATOL = 1e-3
LOOSE_RTOL = 1e-3


# --------------------------------------------------------------------------- #
# Synthetic input generators                                                   #
# --------------------------------------------------------------------------- #
def seed_rng(seed: int = SEED):
    """Seed torch's CPU + CUDA RNGs deterministically."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def gen_bev(batch: int = 2, channels: int = 32, h: int = 320, w: int = 320,
            *, seed: int = SEED, device='cpu'):
    """Return a fixed-seed ``(B, C, H, W)`` BEV feature tensor."""
    seed_rng(seed)
    return torch.randn(batch, channels, h, w, device=device)


def gen_multiscale(shapes, *, seed: int = SEED, device='cpu'):
    """Return a list of fixed-seed tensors with the given shapes."""
    seed_rng(seed)
    return [torch.randn(*s, device=device) for s in shapes]


def gen_gt_boxes(batch: int = 2, max_objs: int = 8, *, seed: int = SEED,
                 pc_range=(0.0, -25.6, -3.0, 51.2, 25.6, 2.0), device='cpu'):
    """Random ``gt_boxes`` tensor (B, M, 8) with 1-based class labels.

    Boxes are kept inside ``pc_range`` so the head's range filter doesn't drop
    all of them. Returns float32 [x, y, z, dx, dy, dz, heading, class].
    """
    seed_rng(seed)
    x0, y0, z0, x1, y1, z1 = pc_range
    cx = torch.empty(batch, max_objs, device=device).uniform_(x0 + 2, x1 - 2)
    cy = torch.empty(batch, max_objs, device=device).uniform_(y0 + 2, y1 - 2)
    cz = torch.empty(batch, max_objs, device=device).uniform_(z0 + 0.5, z1 - 0.5)
    dx = torch.empty(batch, max_objs, device=device).uniform_(1.0, 4.0)
    dy = torch.empty(batch, max_objs, device=device).uniform_(0.5, 1.8)
    dz = torch.empty(batch, max_objs, device=device).uniform_(1.0, 1.8)
    rot = torch.empty(batch, max_objs, device=device).uniform_(-3.14, 3.14)
    cls = torch.randint(1, 4, (batch, max_objs), device=device).float()
    return torch.stack([cx, cy, cz, dx, dy, dz, rot, cls], dim=-1)


# --------------------------------------------------------------------------- #
# Weight alignment (premise #1)                                                #
# --------------------------------------------------------------------------- #
def align_state_dicts(src_module, dst_module, *, strict_key_match=True,
                      verbose=False):
    """Copy every tensor from ``src_module.state_dict()`` into ``dst_module``
    whose key + shape matches exactly.

    This is the premise-#1 weight alignment: BOTH sides are constructed with
    ``torch.manual_seed(0)`` then we additionally copy by name so the weights
    are guaranteed identical regardless of any RNG subtleties.

    Returns a report dict with keys:
        ``matched``  : list[(key, shape)]
        ``unmatched_src`` : list[(key, shape)]  keys only in src
        ``unmatched_dst`` : list[(key, shape)]  keys only in dst
        ``shape_mismatch``: list[(key, src_shape, dst_shape)]
    """
    src_sd = src_module.state_dict()
    dst_sd = dst_module.state_dict()

    matched, unmatched_src, unmatched_dst, shape_mismatch = [], [], [], []
    new_sd = {}
    for k, v in dst_sd.items():
        if k in src_sd:
            if src_sd[k].shape == v.shape:
                new_sd[k] = src_sd[k].clone()
                matched.append((k, tuple(v.shape)))
            else:
                new_sd[k] = v  # leave as-is
                shape_mismatch.append((k, tuple(src_sd[k].shape), tuple(v.shape)))
        else:
            new_sd[k] = v
            unmatched_dst.append((k, tuple(v.shape)))
    for k, v in src_sd.items():
        if k not in dst_sd:
            unmatched_src.append((k, tuple(v.shape)))

    dst_module.load_state_dict(new_sd, strict=False)

    report = {
        'matched': matched,
        'unmatched_src': unmatched_src,
        'unmatched_dst': unmatched_dst,
        'shape_mismatch': shape_mismatch,
    }
    if verbose:
        print(f'  [align] matched={len(matched)} '
              f'unmatched_src={len(unmatched_src)} '
              f'unmatched_dst={len(unmatched_dst)} '
              f'shape_mismatch={len(shape_mismatch)}')
        for k, s, d in shape_mismatch:
            print(f'    SHAPE MISMATCH {k}: src={s} dst={d}')
    return report


# --------------------------------------------------------------------------- #
# allclose assertor with drilldown                                             #
# --------------------------------------------------------------------------- #
def parity_allclose(a, b, *, atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL,
                    name='output', verbose=True):
    """Assert ``a`` and ``b`` are elementwise close.

    On failure, locates the first mismatched index and prints its two values
    plus the relative error. Returns ``(passed, max_abs_diff, max_rel_diff)``
    so callers can record per-point diagnostics even when the test passes.
    """
    a = a.detach()
    b = b.detach()
    if a.shape != b.shape:
        msg = (f'[{name}] SHAPE MISMATCH: {tuple(a.shape)} vs {tuple(b.shape)}')
        if verbose:
            print(msg)
        return False, float('inf'), float('inf')

    close = torch.isclose(a, b, atol=atol, rtol=rtol)
    passed = bool(close.all().item())
    abs_diff = (a - b).abs()
    max_abs = float(abs_diff.max().item())
    # Rel diff w.r.t. |b|+eps to avoid div-by-zero.
    rel = abs_diff / (b.abs() + 1e-12)
    max_rel = float(rel.max().item())

    if not passed and verbose:
        mismatch = (~close).flatten()
        first_flat = int(mismatch.nonzero(as_tuple=False)[0].item())
        # unravel
        idx = []
        rest = first_flat
        for dim in reversed(a.shape):
            idx.append(rest % dim)
            rest //= dim
        idx = tuple(reversed(idx))
        av = float(a[idx].item())
        bv = float(b[idx].item())
        print(f'[{name}] FAIL at idx {idx}: a={av:.6e} b={bv:.6e} '
              f'abs_diff={abs_diff[idx].item():.6e} '
              f'rel_diff={rel[idx].item():.6e} '
              f'(atol={atol} rtol={rtol})')
        print(f'  overall max_abs_diff={max_abs:.6e} max_rel_diff={max_rel:.6e}')
    elif verbose:
        print(f'[{name}] PASS  max_abs_diff={max_abs:.3e} '
              f'max_rel_diff={max_rel:.3e} (atol={atol} rtol={rtol})')

    return passed, max_abs, max_rel


def parity_allclose_list(a_list, b_list, *, atol=DEFAULT_ATOL,
                         rtol=DEFAULT_RTOL, name='output', verbose=True):
    """Same as parity_allclose but for lists of tensors (multi-scale)."""
    assert len(a_list) == len(b_list), \
        f'[{name}] list length mismatch: {len(a_list)} vs {len(b_list)}'
    all_pass = True
    max_abs = 0.0
    max_rel = 0.0
    for i, (a, b) in enumerate(zip(a_list, b_list)):
        p, ma, mr = parity_allclose(
            a, b, atol=atol, rtol=rtol,
            name=f'{name}[{i}]', verbose=verbose)
        all_pass = all_pass and p
        max_abs = max(max_abs, ma)
        max_rel = max(max_rel, mr)
    return all_pass, max_abs, max_rel
