"""Small, explicit forward/backward probe on the four allocated GPUs."""
import argparse
import json
from pathlib import Path


def check():
    import torch
    import torch.nn.functional as F
    from causal_conv1d import causal_conv1d_fn
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from science_gspo_g.model import training_kernel_audit
    audit = training_kernel_audit()
    assert torch.cuda.device_count() == 4, 'Probe requires the four allocated GPUs'
    records = []
    for device in range(4):
        torch.cuda.set_device(device)
        torch.manual_seed(100 + device)
        x = torch.randn(2, 128, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
        w = torch.randn(128, 4, device=device, dtype=torch.bfloat16, requires_grad=True)
        conv = causal_conv1d_fn(x, w, activation='silu')
        reference = F.silu(F.conv1d(x.float(), w.float().unsqueeze(1), padding=3, groups=128)[..., :64])
        assert torch.allclose(conv.float(), reference, rtol=0.03, atol=0.03), 'Causal convolution parity failed'
        conv.float().square().mean().backward()
        assert all(torch.isfinite(t.grad).all() and t.grad.abs().sum() > 0 for t in (x, w))
        q, k, v = [torch.randn(1, 128, 4, 128, device=device, dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
        g = -F.softplus(torch.randn(1, 128, 4, device=device)) * 0.05
        beta = torch.sigmoid(torch.randn(1, 128, 4, device=device, dtype=torch.bfloat16))
        out, _ = chunk_gated_delta_rule(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True)
        assert torch.isfinite(out).all(), 'Nonfinite DeltaNet output'
        out.float().square().mean().backward()
        assert all(torch.isfinite(t.grad).all() and t.grad.abs().sum() > 0 for t in (q, k, v))
        torch.cuda.synchronize(device)
        records.append(dict(device=device, gpu=torch.cuda.get_device_name(device),
                            convolution_max_absolute_error=float((conv.float()-reference).abs().max()),
                            deltanet_gradient_norms=[float(t.grad.float().norm()) for t in (q, k, v)],
                            convolution_backward=type(conv.grad_fn).__name__, deltanet_backward=type(out.grad_fn).__name__,
                            finite_nonzero_gradients=True))
        del x, w, conv, reference, q, k, v, g, beta, out
        torch.cuda.empty_cache()
    return dict(success=True, kernels=audit, devices=records,
                limitation='Small kernel probe; the policy pilot still must verify full-sequence execution and timing.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    args = p.parse_args()
    result = check()
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
