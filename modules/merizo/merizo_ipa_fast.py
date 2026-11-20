"""Inference-only exact-order shortcuts for the shared six-block Merizo IPA."""
import ctypes
from functools import lru_cache
import math
from pathlib import Path

import torch

from modules.merizo.runtime import supports_fast_path


@lru_cache(maxsize=1)
def native_points():
    filename = Path(__file__).with_name('merizo_points.so')
    if not filename.is_file():
        return None
    try:
        library = ctypes.CDLL(str(filename))
    except OSError:
        return None
    kernel = library.merizo_point_logits
    kernel.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
    kernel.restype = ctypes.c_int
    return kernel


def point_logits(q, k, weights):
    batch, length, heads = q.shape[:3]
    q, k = q.contiguous(), k.contiguous()
    result = q.new_empty((batch, length, length, heads))
    with torch.cuda.device(q.device):
        status = native_points()(q.data_ptr(), k.data_ptr(), weights.data_ptr(), result.data_ptr(),
                                 batch, length, heads, torch.cuda.current_stream().cuda_stream)
    if status:
        raise RuntimeError(f'Merizo point-distance CUDA launch returned {status}')
    return result


@lru_cache(maxsize=1)
def native_values():
    kernel = ctypes.CDLL(str(Path(__file__).with_name('merizo_points.so'))).merizo_weighted_points
    kernel.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
    kernel.restype = ctypes.c_int
    return kernel


def weighted_values(a, values):
    batch, heads, length, _ = a.shape
    if (batch != 1 or heads != 16 or length > 1400 or values.shape[-2:] != (8, 3)
            or not a.is_contiguous() or not supports_fast_path(a.device)):
        values = values.permute(0, 2, 4, 1, 3)[..., None, :, :]
        return torch.sum(a[..., None, :, :, None] * values, dim=-2)
    order = (0, 2, 1, 3, 4) if length < 256 else (0, 2, 3, 4, 1)
    values = values.permute(order).contiguous()
    result = a.new_empty((batch, heads, 3, length, 8))
    with torch.cuda.device(a.device):
        status = native_values()(a.data_ptr(), values.data_ptr(), result.data_ptr(),
                                 batch, length, heads, torch.cuda.current_stream().cuda_stream)
    if status:
        raise RuntimeError(f'Merizo weighted point CUDA launch returned {status}')
    return result


def attention_forward(module, s, z, r, constants):
    """Follow the pinned scalar, point and pair contractions in their original order."""
    from model.ipa.nndef_ipa_primitives import flatten_final_dims, permute_final_dims

    bias, square_mask, weights = constants
    heads = module.no_heads
    q = module.linear_q(s).reshape(*s.shape[:-1], heads, module.c_hidden)
    kv = module.linear_kv(s).reshape(*s.shape[:-1], heads, 2 * module.c_hidden)
    k, v = kv.split(module.c_hidden, dim=-1)

    def global_points(projected):
        xyz = torch.stack(projected.chunk(3, dim=-1), dim=-1)
        xyz = r[..., None].apply(xyz)
        return xyz.reshape(*s.shape[:-1], heads, -1, 3)

    q_pts = global_points(module.linear_q_points(s))
    k_pts, v_pts = global_points(module.linear_kv_points(s)).split(
        [module.no_qk_points, module.no_v_points], dim=-2)
    q = module.rotary_emb.rotate_queries_or_keys(q)
    k = module.rotary_emb.rotate_queries_or_keys(k)
    a = torch.matmul(permute_final_dims(q, (1, 0, 2)), permute_final_dims(k, (1, 2, 0)))
    a *= math.sqrt(1.0 / (3 * module.c_hidden))
    a += bias
    a = a + permute_final_dims(point_logits(q_pts, k_pts, weights), (2, 0, 1))
    a = module.softmax(a + square_mask)

    scalar = torch.matmul(a, v.transpose(-2, -3).to(dtype=a.dtype)).transpose(-2, -3)
    points = weighted_values(a, v_pts)
    points = r[..., None, None].invert_apply(permute_final_dims(points, (2, 0, 3, 1)))
    norms = flatten_final_dims(torch.sqrt(torch.sum(points ** 2, dim=-1) + module.eps), 2)
    points = points.reshape(*points.shape[:-3], -1, 3)
    pairs = torch.matmul(a.transpose(-2, -3), z.to(dtype=a.dtype))
    output = module.hidden_out(flatten_final_dims(scalar, 2))
    output += module.pair_out(flatten_final_dims(pairs, 2))
    output += module.points_out(torch.cat(torch.unbind(points, dim=-1), dim=-1))
    output += module.points_norm_out(norms)
    return output, a


def install_merizo_ipa():
    from model.ipa.ipa_encoder import ipa_block
    from model.ipa.nndef_ipa import InvariantPointAttention
    from model.ipa.nndef_ipa_primitives import permute_final_dims

    previous_block = ipa_block.forward
    previous_attention = InvariantPointAttention.forward

    def attention(module, s, z, r, mask, constants=None):
        if constants is None:
            return previous_attention(module, s, z, r, mask)
        return attention_forward(module, s, z, r, constants)

    def block(module, s, z, rigids, mask=None):
        ipa = module.ipa
        if (module.training or torch.is_grad_enabled() or torch.is_autocast_enabled()
                or not s.is_cuda or s.dtype != torch.float32
                or not supports_fast_path(s.device) or s.ndim != 3 or ipa.no_qk_points != 4 or not 0 < s.shape[1] <= 65535
                or torch.cuda.get_device_capability(s.device)[0] < 8 or native_points() is None):
            return previous_block(module, s, z, rigids, mask)
        s = module.linear_in(module.layer_norm_s(s))
        z = module.layer_norm_z(z)
        mask = s.new_ones(s.shape[:-1]) if mask is None else mask
        square_mask = (ipa.inf * (mask.unsqueeze(-1) * mask.unsqueeze(-2) - 1)).unsqueeze(-3)
        bias = math.sqrt(1.0 / 3) * permute_final_dims(ipa.linear_b(z), (2, 0, 1))
        weights = ipa.softplus(ipa.head_weights) * math.sqrt(1.0 / (3 * (ipa.no_qk_points * 9.0 / 2)))
        for _ in range(module.no_blocks):
            update, attn = ipa(s=s, z=z, r=rigids, mask=mask, constants=(bias, square_mask, weights))
            s = module.transition(module.layer_norm_ipa(module.ipa_dropout(s + update)))
        return s, attn

    InvariantPointAttention.forward = attention
    ipa_block.forward = block
