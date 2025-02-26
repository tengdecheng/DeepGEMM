import random
import torch
from typing import Tuple
import triton

import deep_gemm
from deep_gemm import bench_kineto, calc_diff, cell_div, get_col_major_tma_aligned_tensor

from sgl_kernel import fp8_blockwise_scaled_mm as sglang_fp8_blockwise_scaled_mm
from vllm._custom_ops import cutlass_scaled_mm as vllm_fp8_blockwise_scaled_mm


def per_token_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2 and x.size(1) % 128 == 0
    m, n = x.shape
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    return (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, n), (x_amax / 448.0).view(m, -1)

def per_block_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros((cell_div(m, 128) * 128, cell_div(n, 128) * 128), dtype=x.dtype, device=x.device)
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (x_amax / 448.0).view(x_view.size(0), x_view.size(2))

def construct(m: int, k: int, n: int) -> \
        Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
    x = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
    y = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)
    out = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
    ref_out = x @ y.t()

    x_fp8, y_fp8 = per_token_cast_to_fp8(x), per_block_cast_to_fp8(y)
    # Transpose earlier so that the testing will not trigger transposing kernels
    x_fp8 = (x_fp8[0], get_col_major_tma_aligned_tensor(x_fp8[1]))
    return x_fp8, y_fp8, out, ref_out

##########################

def scale_shape(shape, group_shape):
    assert len(shape) == len(group_shape)
    return tuple(triton.cdiv(shape[i], group_shape[i]) for i in range(len(group_shape)))

def construct_sglang_vllm(m: int, k: int, n: int) -> \
        Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
    out = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
    
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max, fp8_min = fp8_info.max, fp8_info.min

    a_fp32 = (torch.rand(m, k, dtype=torch.float32, device="cuda") - 0.5) * 2 * fp8_max
    a_fp8 = a_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

    b_fp32 = (torch.rand(n, k, dtype=torch.float32, device="cuda") - 0.5) * 2 * fp8_max
    b_fp8 = b_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn).t()

    scale_a_group_shape = (1, 128)
    scale_b_group_shape = (128, 128)
    scale_a_shape = scale_shape(a_fp8.shape, scale_a_group_shape)
    scale_b_shape = scale_shape(b_fp8.shape, scale_b_group_shape)

    scale_a = torch.randn(scale_a_shape, device="cuda", dtype=torch.float32)
    scale_b = torch.randn(scale_b_shape, device="cuda", dtype=torch.float32)
    scale_a = scale_a.t().contiguous().t()
    scale_b = scale_b.t().contiguous().t()
    
    x_fp8 = (a_fp8, scale_a)
    y_fp8 = (b_fp8, scale_b)
    return x_fp8, y_fp8, out, out

def bench_sglang_gemm(fn) -> None:
    print('Testing sglang cutlass GEMM:')
    for m in (64, 128, 4096):
        for k, n in [(7168, 2112), (1536, 24576), (512, 32768), (16384, 7168), (7168, 4096), (2048, 7168)]:
            k, n = triton.cdiv(k, 128) * 128, triton.cdiv(n, 128) * 128
            (x_fp8, x_fp8_scales), (y_fp8, y_fp8_scales), out, ref_out = construct_sglang_vllm(m, k, n)
            out = sglang_fp8_blockwise_scaled_mm(x_fp8, y_fp8, x_fp8_scales, y_fp8_scales, out_dtype=torch.bfloat16)

            def test_func():
                out = sglang_fp8_blockwise_scaled_mm(x_fp8, y_fp8, x_fp8_scales, y_fp8_scales, out_dtype=torch.bfloat16)

            t = deep_gemm.bench(test_func, num_tests=50)
            tflops = 2 * m * n * k / (t * 1e-3) / 1e12
            bandwidth = (m * k + k * n + m * n * 2) / 1e9 / (t * 1e-3)
            print(f' > Performance (m={m:5}, n={n:5}, k={k:5}): {t * 1e3:4.0f} us | '
                  f'throughput: {tflops:4.0f} TFLOPS, '
                  f'{bandwidth:4.0f} GB/s')
            
            fn(name="sglang_gemm_fp8",m=m,n=n,k=k,tflops=tflops,bw=bandwidth)
    print()
    
def bench_vllm_gemm(fn) -> None:
    print('Testing vllm cutlass GEMM:')
    for m in (64, 128, 4096):
        for k, n in [(7168, 2112), (1536, 24576), (512, 32768), (16384, 7168), (7168, 4096), (2048, 7168)]:
            k, n = triton.cdiv(k, 128) * 128, triton.cdiv(n, 128) * 128
            (x_fp8, x_fp8_scales), (y_fp8, y_fp8_scales), out, ref_out = construct_sglang_vllm(m, k, n)
            out = vllm_fp8_blockwise_scaled_mm(x_fp8, y_fp8, x_fp8_scales, y_fp8_scales, out_dtype=torch.bfloat16)
            
            # noinspection PyShadowingNames
            def test_func():
                out = vllm_fp8_blockwise_scaled_mm(x_fp8, y_fp8, x_fp8_scales, y_fp8_scales, out_dtype=torch.bfloat16)

            t = deep_gemm.bench(test_func, num_tests=50)
            tflops = 2 * m * n * k / (t * 1e-3) / 1e12
            bandwidth = (m * k + k * n + m * n * 2) / 1e9 / (t * 1e-3)
            print(f' > Performance (m={m:5}, n={n:5}, k={k:5}): {t * 1e3:4.0f} us | '
                  f'throughput: {tflops:4.0f} TFLOPS, '
                  f'{bandwidth:4.0f} GB/s')
            
            fn(name="vllm_gemm_fp8",m=m,n=n,k=k,tflops=tflops,bw=bandwidth)
    print()

def bench_deep_gemm(fn) -> None:
    print('Testing Deep GEMM:')
    for m in (64, 128, 4096):
        for k, n in [(7168, 2112), (1536, 24576), (512, 32768), (16384, 7168), (7168, 4096), (2048, 7168)]:
            k, n = triton.cdiv(k, 128) * 128, triton.cdiv(n, 128) * 128
            
            x_fp8, y_fp8, out, ref_out = construct(m, k, n)
            deep_gemm.gemm_fp8_fp8_bf16_nt(x_fp8, y_fp8, out)
            diff = calc_diff(out, ref_out)
            assert diff < 0.001, f'{m=}, {k=}, {n=}, {diff:.5f}'

            def test_func():
                deep_gemm.gemm_fp8_fp8_bf16_nt(x_fp8, y_fp8, out)

            t = deep_gemm.bench(test_func, num_tests=50)
            
            tflops = 2 * m * n * k / (t * 1e-3) / 1e12
            bandwidth = (m * k + k * n + m * n * 2) / 1e9 / (t * 1e-3)
            print(f' > Performance (m={m:5}, n={n:5}, k={k:5}): {t * 1e3:4.0f} us | '
                  f'throughput: {tflops:4.0f} TFLOPS, '
                  f'{bandwidth:4.0f} GB/s')
            
            fn(name="deep_gemm_fp8",m=m,n=n,k=k,tflops=tflops,bw=bandwidth)
    print()

if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(0)
    random.seed(0)

    print('Library path:')
    print(f' > {deep_gemm.__path__}\n')
    
    f = open("perf.csv", "w")
    f.write("name,m,n,k,tflops,bw\n")
    def write_to_csv(**args):
        name = args["name"]
        m = args["m"]
        n = args["n"]
        k = args["k"]
        tflops = args["tflops"]
        bw = args["bw"]
        f.write(f"{name},{m},{n},{k},{tflops:.2f},{bw:.2f}\n")

    bench_deep_gemm(write_to_csv)
    bench_sglang_gemm(write_to_csv)
    bench_vllm_gemm(write_to_csv)