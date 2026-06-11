import math
from enum import Enum
from dataclasses import dataclass

import torch


torch.manual_seed(7)


class WeightFormat(Enum):
    U8 = "u8"
    I8 = "i8"
    U4 = "u4"
    I4 = "i4"


def is_nibble_format(weight_format: WeightFormat):
    return weight_format in (WeightFormat.U4, WeightFormat.I4)


@dataclass
class SDQConfig:
    m: int = 4
    k: int = 128
    n: int = 32
    group_size: int = 16
    weight_format: WeightFormat = WeightFormat.U8


def weight_value_range(weight_format: WeightFormat):
    if weight_format == WeightFormat.U8:
        return 0, 255, True
    if weight_format == WeightFormat.I8:
        return -128, 127, False
    if weight_format == WeightFormat.U4:
        return 0, 15, True
    if weight_format == WeightFormat.I4:
        return -8, 7, False
    raise ValueError(f"Unsupported weight format: {weight_format}")


def quantize_weights_per_group(wei, group_size, weight_format: WeightFormat):
    k, n = wei.shape
    group_count = k // group_size
    qmin, qmax, is_unsigned = weight_value_range(weight_format)

    weight_scales = torch.empty(group_count, n, dtype=torch.float32)
    if is_unsigned:
        zp_max = min(qmax, 15)
        zero_points = torch.empty(group_count, n, dtype=torch.float32)
    else:
        zero_points = torch.zeros(group_count, n, dtype=torch.float32)

    qweight = torch.empty(k, n, dtype=torch.int32)

    for group in range(group_count):
        begin = group * group_size
        end = begin + group_size
        wei_group = wei[begin:end, :]

        for col in range(n):
            wei_col = wei_group[:, col]
            wmin = wei_col.min().item()
            wmax = wei_col.max().item()

            if is_unsigned:
                scale = (wmax - wmin) / qmax if wmax != wmin else 1.0
                zp = -round(wmin / scale) if scale != 0.0 else 0
                zp = max(0, min(zp, zp_max))
            else:
                amax = max(abs(wmin), abs(wmax))
                scale = amax / qmax if amax != 0.0 else 1.0
                zp = 0

            weight_scales[group, col] = scale
            zero_points[group, col] = zp

            if scale != 0.0:
                qweight[begin:end, col] = torch.clamp(
                    torch.round(wei_col / scale) + zp, qmin, qmax
                ).to(torch.int32)
            else:
                qweight[begin:end, col] = zp

    return qweight, weight_scales, zero_points


def make_random_problem(cfg: SDQConfig):
    assert cfg.k % cfg.group_size == 0

    src = torch.randn(cfg.m, cfg.k, dtype=torch.float32)
    wei = torch.randn(cfg.k, cfg.n, dtype=torch.float32)

    qwei, wei_s, wei_zp = quantize_weights_per_group(wei, cfg.group_size, cfg.weight_format)

    bias = torch.randn(cfg.n, dtype=torch.float32)
    return src, qwei, wei_s, wei_zp, bias


# =========================
# Weight Side
# =========================
def pack_nibbles(qweight_i32: torch.Tensor) -> torch.Tensor:
    flat = qweight_i32.reshape(-1).to(torch.int32)
    if flat.numel() % 2 != 0:
        raise ValueError("Nibble packing requires an even number of values")
    low = flat[0::2]
    high = flat[1::2]
    packed = (low & 0x0F) | ((high & 0x0F) << 4)
    return packed.to(torch.uint8)


def unpack_nibbles(packed: torch.Tensor, shape, signed: bool) -> torch.Tensor:
    packed_i32 = packed.to(torch.int32)
    low = packed_i32 & 0x0F
    high = (packed_i32 >> 4) & 0x0F
    unpacked = torch.empty(packed.numel() * 2, dtype=torch.int32)
    unpacked[0::2] = low
    unpacked[1::2] = high
    if signed:
        unpacked = torch.where(unpacked >= 8, unpacked - 16, unpacked)
    return unpacked.view(*shape)


def materialize_logical_qweight(qweight, weight_format: WeightFormat):
    if weight_format == WeightFormat.U8 or weight_format == WeightFormat.I8:
        return qweight.to(torch.int32)
    if weight_format == WeightFormat.U4:
        packed = pack_nibbles(qweight)
        return unpack_nibbles(packed, qweight.shape, signed=False)
    if weight_format == WeightFormat.I4:
        packed = pack_nibbles(qweight & 0x0F)
        return unpack_nibbles(packed, qweight.shape, signed=True)
    raise ValueError(f"Unsupported weight format: {weight_format}")


def dequantize_weights(qweight, weight_scales, zero_points, group_size):
    k, n = qweight.shape
    group_count = k // group_size
    qweight_f32 = qweight.to(torch.float32)
    weight_f32 = torch.empty(k, n, dtype=torch.float32)

    for group in range(group_count):
        begin = group * group_size
        end = begin + group_size
        weight_f32[begin:end] = (qweight_f32[begin:end] - zero_points[group].view(1, n)) * weight_scales[group].view(1, n)

    return weight_f32


def float_reference(src, qweight, weight_scales, zero_points, bias, group_size, weight_format):
    logical_qweight = materialize_logical_qweight(qweight, weight_format)
    weights = dequantize_weights(logical_qweight, weight_scales, zero_points, group_size)
    return src @ weights + bias


# =========================
# Source Side
# =========================
def quantize_source_dynamic(src, group_size):
    m, k = src.shape
    group_count = math.ceil(k / group_size)
    qsrc = torch.empty(m, k, dtype=torch.int8)
    src_scales = torch.empty(m, group_count, dtype=torch.float32)
    grouped_sums = torch.empty(m, group_count, dtype=torch.int32)

    for row in range(m):
        for group in range(group_count):
            begin = group * group_size
            end = min(begin + group_size, k)
            block = src[row, begin:end]
            amax = block.abs().max().item() if block.numel() > 0 else 0.0
            dscale = amax / 127.0 if amax != 0.0 else 0.0
            src_scales[row, group] = dscale

            if dscale == 0.0:
                qblock = torch.zeros_like(block, dtype=torch.int8)
            else:
                qblock = torch.clamp(torch.round(block / dscale), -127, 127).to(torch.int8)

            qsrc[row, begin:end] = qblock
            grouped_sums[row, group] = qblock.to(torch.int32).sum()

    return qsrc, src_scales, grouped_sums


# =========================
# Integer Core + Finalize Side
# =========================
def sdq_matmul(src, qweight, weight_scales, zero_points, bias, group_size, weight_format):
    m, k = src.shape
    _, n = qweight.shape
    group_count = math.ceil(k / group_size)

    qsrc, src_scales, grouped_sums = quantize_source_dynamic(src, group_size)
    output = torch.zeros(m, n, dtype=torch.float32)

    qweight_i32 = materialize_logical_qweight(qweight, weight_format)

    for group in range(group_count):
        begin = group * group_size
        end = min(begin + group_size, k)

        qsrc_group = qsrc[:, begin:end].to(torch.int32)
        qweight_group = qweight_i32[begin:end, :]

        dot = qsrc_group @ qweight_group
        scale = src_scales[:, group].view(m, 1) * weight_scales[group].view(1, n)
        compensation = grouped_sums[:, group].to(torch.float32).view(m, 1) * zero_points[group].view(1, n) * scale

        output += dot.to(torch.float32) * scale - compensation

    output += bias.view(1, n)
    return output, qsrc, src_scales, grouped_sums


def summarize(name, reference, candidate):
    diff = candidate - reference
    abs_diff = diff.abs()
    print(f"[{name}] max abs diff: {abs_diff.max().item():.8f}")
    print(f"[{name}] mean abs diff: {abs_diff.mean().item():.8f}")
    print(f"[{name}] rmse: {torch.sqrt((diff * diff).mean()).item():.8f}")


def run_case(weight_dtype):
    cfg = SDQConfig(weight_format=weight_dtype)
    src, qweight, weight_scales, zero_points, bias = make_random_problem(cfg)

    logical_qweight = materialize_logical_qweight(qweight, cfg.weight_format)
    reference = float_reference(src, qweight, weight_scales, zero_points, bias, cfg.group_size, cfg.weight_format)
    candidate, qsrc, src_scales, grouped_sums = sdq_matmul(src,
                                                           qweight,
                                                           weight_scales,
                                                           zero_points,
                                                           bias,
                                                           cfg.group_size,
                                                           cfg.weight_format)

    print("=" * 80)
    print(f"Case: weight_format={weight_dtype.value}")
    print(f"src shape={tuple(src.shape)}, qweight shape={tuple(qweight.shape)}, group_size={cfg.group_size}")
    print(f"qsrc dtype={qsrc.dtype}, src_scales shape={tuple(src_scales.shape)}, grouped_sums shape={tuple(grouped_sums.shape)}")
    if is_nibble_format(cfg.weight_format):
        packed = pack_nibbles((qweight & 0x0F) if cfg.weight_format == WeightFormat.I4 else qweight)
        print(f"packed nibble bytes shape={tuple(packed.shape)}, logical_qweight dtype={logical_qweight.dtype}")
    summarize(weight_dtype.value, reference, candidate)
    print("reference[0, :8] =", reference[0, :8])
    print("candidate[0, :8] =", candidate[0, :8])


if __name__ == "__main__":
    run_case(WeightFormat.U8)
    run_case(WeightFormat.I8)
    run_case(WeightFormat.U4)
    run_case(WeightFormat.I4)
