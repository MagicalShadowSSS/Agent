"""Preallocate outputs of already token-chunked MLP/RMSNorm instead of retaining all parts plus concat."""
def mlp(x, forward, chunk=2048):
    if x.shape[-2] <= chunk:
        return forward(x)
    first = forward(x.narrow(-2, 0, chunk))
    shape = list(first.shape)
    shape[-2] = x.shape[-2]
    result = first.new_empty(shape)
    result.narrow(-2, 0, chunk).copy_(first)
    del first
    for start in range(chunk, x.shape[-2], chunk):
        size = min(chunk, x.shape[-2] - start)
        value = forward(x.narrow(-2, start, size))
        result.narrow(-2, start, size).copy_(value)
        del value
    return result


def norm(x, forward, chunk=2048):
    rows = x.reshape(-1, x.shape[-1])
    if rows.shape[0] <= chunk:
        return forward(x)
    first = forward(rows[:chunk])
    result = first.new_empty(rows.shape)
    result[:chunk].copy_(first)
    del first
    for start in range(chunk, rows.shape[0], chunk):
        end = min(start + chunk, rows.shape[0])
        value = forward(rows[start:end])
        result[start:end].copy_(value)
        del value
    return result.reshape(x.shape)
