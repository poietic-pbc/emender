import torch

import ndm.triton.e88_triton_optimized as optimized


def test_unaligned_forward_passes_real_state_boundary(monkeypatch):
    captured = {}

    def fake_triton(S0, k, v, q, decay, g=None, **kwargs):
        captured.update(
            T=k.shape[0],
            valid_length=kwargs["valid_length"],
            padded_decay=decay[5:].clone(),
        )
        out = torch.zeros(
            (k.shape[0], k.shape[1], k.shape[2], v.shape[-1]),
            dtype=v.dtype,
        )
        return out, S0.clone()

    monkeypatch.setattr(optimized, "e88_triton", fake_triton)

    B, T, H, N, V = 1, 5, 2, 4, 3
    S0 = torch.randn(B, H, N, V)
    k = torch.randn(B, T, H, N)
    v = torch.randn(B, T, H, V)
    q = torch.randn(B, T, H, N)
    decay = torch.rand(B, T, H)
    erase = torch.rand(B, T, H, N)
    write = torch.rand(B, T, H, V)

    state, out = optimized.e88_triton_optimized_apply(
        False, k, v, q, decay, None, S0, H,
        False, False, 16,
        erase_gate=erase,
        value_write_gate=write,
    )

    assert captured["T"] == 16
    assert captured["valid_length"] == T
    assert torch.count_nonzero(captured["padded_decay"]) == 0
    assert out.shape == (B, T, H, V)
    torch.testing.assert_close(state, S0)
