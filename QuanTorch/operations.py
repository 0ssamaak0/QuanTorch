import torch
# Basic Operations
def inner_product(bra, ket):
    """
    Calculates the inner product of two vectors using Dirac notation.

    Args:
        bra (torch.Tensor): the Bra Vector
        ket (torch.Tensor): the Ket Vector (before conjugation)
    """
    return torch.conj(bra).T @ ket

def outer_product(ket, bra):
    """
    Calculates the outer product of two vectors using Dirac notation.

    Args:
        ket (torch.Tensor): the Ket Vector
        bra (torch.Tensor): the Bra Vector (before conjugation)
    """
    return ket @ torch.conj(bra).T

def tensor_product(ket, bra):
    """
    Calculates the tensor product of two vectors using Dirac notation.

    Args:
        ket (torch.Tensor): the Ket Vector
        bra (torch.Tensor): the Bra Vector (before conjugation)
    """
    return torch.kron(ket, bra)