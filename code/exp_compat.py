"""torch_scatter compatibility shim for DrugMiR.

Import BEFORE hp_finetune. If the real torch_scatter is installed this module
does nothing at all, so GPU runs keep using the original package.
"""
import sys, types
import torch

def scatter_mean(src, index, dim=0, dim_size=None):
    if dim_size is None:
        dim_size = int(index.max().item()) + 1 if index.numel() else 0
    shape = list(src.shape); shape[dim] = dim_size
    out = torch.zeros(shape, dtype=src.dtype, device=src.device)
    idx = index
    while idx.dim() < src.dim():
        idx = idx.unsqueeze(-1)
    out.scatter_add_(dim, idx.expand_as(src), src)
    cnt = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
    cnt.scatter_add_(0, index, torch.ones_like(index, dtype=src.dtype))
    view = [1] * src.dim(); view[dim] = dim_size
    return out / cnt.clamp(min=1).view(view)

def _install():
    try:
        import torch_scatter  # noqa: F401
        return False
    except ImportError:
        pass
    m = types.ModuleType('torch_scatter')
    m.scatter_mean = scatter_mean
    sys.modules['torch_scatter'] = m
    return True

SHIM_ACTIVE = _install()
if SHIM_ACTIVE:
    print('[exp_compat] torch_scatter missing -> using built-in shim', flush=True)

if __name__ == '__main__':
    torch.manual_seed(0)
    src = torch.randn(200, 7); index = torch.randint(0, 12, (200,))
    got = scatter_mean(src, index, dim=0, dim_size=15)
    ref = torch.zeros(15, 7)
    for g in range(15):
        m = index == g
        if m.any(): ref[g] = src[m].mean(0)
    err = (got - ref).abs().max().item()
    print('shim active      :', SHIM_ACTIVE)
    print('max abs error    : %.3e' % err)
    print('empty buckets = 0:', bool((got[12:] == 0).all()))
    assert err < 1e-5
    print('SELF-TEST PASSED')
