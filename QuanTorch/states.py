import torch


# define a tensor class with complex64 dtype
def qstate(*args, **kwargs):
    tensor = torch.tensor(*args, **kwargs, dtype=torch.complex64)
    # if it's a vector, make it a column vector
    if len(tensor.shape) == 1:
        tensor = tensor.view(-1, 1)
    return tensor


def density_matrix(*args, **kwargs):
    return torch.tensor(*args, **kwargs, dtype=torch.complex64)
