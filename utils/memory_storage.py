"""Pack scene memory for one GPU-to-CPU transfer without changing loaded values."""
import torch


def store_cpu_layers(layers, dtype):
    result = [None] * len(layers)
    valid = [(i, layer) for i, layer in enumerate(layers) if layer is not None]
    if not valid:
        return result
    first = valid[0][1]["k"]
    uniform = all(layer[key].shape == first.shape and layer[key].device == first.device
                  for _, layer in valid for key in ("k", "v"))
    if first.is_cuda and uniform:
        packed = torch.empty((len(valid), 2, *first.shape), device=first.device, dtype=dtype)
        for j, (_, layer) in enumerate(valid):
            packed[j, 0].copy_(layer["k"])
            packed[j, 1].copy_(layer["v"])
        host = packed.cpu()
        for j, (index, layer) in enumerate(valid):
            result[index] = {**layer, "k": host[j, 0], "v": host[j, 1]}
    else:
        for index, layer in valid:
            result[index] = {**layer,
                             "k": layer["k"].detach().to(dtype=dtype, device="cpu"),
                             "v": layer["v"].detach().to(dtype=dtype, device="cpu")}
    return result
