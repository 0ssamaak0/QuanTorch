import torch
from .basis import *
from .gates import *
from .operations import *
from math import sqrt

# define test function for Deutsch algorithm (constant / balanced)
def test_function(qubit, type = "constant", mode = "a"):
    """
    Test function for Deutsch algorithm, either constant or balanced.
    Args:
        qubit (torch.Tensor): the qubit to apply the gate to
        type (str): either "constant" or "balanced"
    """
    if type == "constant":
        if mode == "a":
            if torch.allclose(qubit, zero):
                return 0
            else:
                return 0
        elif mode == "b":
            if torch.allclose(qubit, zero):
                return 1
            else:
                return 1
    elif type == "balanced":
        if mode == "a":
            if torch.allclose(qubit, zero):
                return 0
            else:
                return 1
        elif mode == "b":
            if torch.allclose(qubit, zero):
                return 1
            else:
                return 0


# Define the Deutsch algorithm using QuanTorch
def deutsch(test_function):
    """
    Deutsch algorithm.
    Args:
        test_function (callable): the test function to use (either constant or balanced)
    """
    q1 = zero
    q2 = zero

    # Apply Hadamard gate to both qubits
    q1 = H(q1)
    q2 = H(q2)

    # Apply test function to both qubits
    