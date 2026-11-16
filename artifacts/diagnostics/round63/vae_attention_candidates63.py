"""Isolated VAE attention candidates. Never imported by the production service.

All eager variants preserve the existing FP16 score/probability rounding.
Padding is an explicit numerical candidate, not assumed bit-exact. The custom
FA variant has different softmax/reduction rounding and is exploratory only.
"""
import torch
from torch.nn import functional as F


def sanitize(x):
    return torch.where(torch.isfinite(x), x, torch.zeros((), dtype=x.dtype, device=x.device))


def eager(q, k, v, *, head_chunk=32, query_chunk=256, pad_query=0, pad_keys=0):
    """BSND I/O, exact legacy operation order, optional shape-only padding."""
    b, n, h, d = q.shape
    assert q.dtype == k.dtype == v.dtype == torch.float16
    assert q.shape == k.shape == v.shape and d == 64 and b == 1
    qq, kk, vv = [x.transpose(1, 2).contiguous() for x in (q, k, v)]
    nq = ((n + pad_query - 1) // pad_query) * pad_query if pad_query else n
    nk = ((n + pad_keys - 1) // pad_keys) * pad_keys if pad_keys else n
    if nq > n:
        qq = F.pad(qq, (0, 0, 0, nq - n))
    if nk > n:
        kk = F.pad(kk, (0, 0, 0, nk - n))
        vv = F.pad(vv, (0, 0, 0, nk - n))
    key_t = kk.transpose(-1, -2).contiguous()
    invalid = torch.arange(nk, device=q.device) >= n if nk > n else None
    heads = []
    for hs in range(0, h, head_chunk):
        qs, ks, vs = [x[:, hs:hs + head_chunk] for x in (qq, key_t, vv)]
        rows = []
        for start in range(0, nq, query_chunk):
            scores = torch.matmul(qs[..., start:start + query_chunk, :], ks) * (d ** -.5)
            if invalid is not None:
                scores = scores.masked_fill(invalid, -50000.)
            scores = scores.clamp(-50000., 50000.)
            p = torch.softmax(scores.float(), dim=-1).to(q.dtype)
            rows.append(torch.matmul(p, vs))
        heads.append(torch.cat(rows, dim=-2)[..., :n, :])
    return sanitize(torch.cat(heads, dim=1).transpose(1, 2).contiguous())


def custom_fa_exploratory(q, k, v):
    from vllm_omni.diffusion.attention.backends import fa910a as fa
    n = q.shape[1]
    nq = ((n + 511) // 512) * 512
    qq, kk, vv = [F.pad(x.transpose(1, 2).contiguous(), (0, 0, 0, nq - n)) for x in (q, k, v)]
    out = fa.flash_attention_910a(qq, kk, vv, n_pad=nq - n)
    return sanitize(out[..., :n, :].transpose(1, 2).contiguous())


def fp32_oracle(q, k, v):
    """Full key set, no half score/probability boundary; not legacy semantics."""
    with torch.autocast('npu', enabled=False):
        qq, kk, vv = [x.transpose(1, 2).float().contiguous() for x in (q, k, v)]
        pieces = []
        for hs in range(0, q.shape[2], 4):
            a, b, c = [x[:, hs:hs + 4] for x in (qq, kk, vv)]
            p = torch.softmax((a @ b.transpose(-1, -2)) * (q.shape[-1] ** -.5), dim=-1)
            pieces.append(p @ c)
        return sanitize(torch.cat(pieces, dim=1).transpose(1, 2).to(q.dtype).contiguous())
