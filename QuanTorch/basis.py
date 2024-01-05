import torch
from math import sqrt

# Define the basis vectors |0> and |1>, and the Hadamard |+> and |-> states

zero = torch.tensor([[1], [0]], dtype=torch.complex64)
one = torch.tensor([[0], [1]], dtype=torch.complex64)
plus = torch.tensor([[1], [1]], dtype=torch.complex64) / sqrt(2)
minus = torch.tensor([[1], [-1]], dtype=torch.complex64) / sqrt(2)

