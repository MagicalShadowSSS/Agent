"""Bound rotary intermediates without changing sequence length or positional math."""
def install(module, chunk_tokens=2048):
    import torch
    original = module.apply_rotary_pos_emb
    def bounded(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        seq_dim = 2 if unsqueeze_dim == 1 else 1
        if unsqueeze_dim not in (1, 2) or q.shape[seq_dim] <= chunk_tokens:
            return original(q, k, cos, sin, position_ids, unsqueeze_dim)
        assert q.shape[seq_dim] == k.shape[seq_dim] == cos.shape[1] == sin.shape[1]
        q_out = torch.empty(q.shape, dtype=torch.promote_types(q.dtype, cos.dtype), device=q.device)
        k_out = torch.empty(k.shape, dtype=torch.promote_types(k.dtype, cos.dtype), device=k.device)
        for start in range(0, q.shape[seq_dim], chunk_tokens):
            end = min(start + chunk_tokens, q.shape[seq_dim])
            axis = [slice(None)] * q.ndim
            axis[seq_dim] = slice(start, end)
            a, b = original(q[tuple(axis)], k[tuple(axis)], cos[:, start:end], sin[:, start:end], position_ids, unsqueeze_dim)
            q_out[tuple(axis)] = a
            k_out[tuple(axis)] = b
            del a, b
        return q_out, k_out
    module.apply_rotary_pos_emb = bounded
    return original, bounded
