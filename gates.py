from helpers import *
import torch


# single qubit gates
def X(qubit):
    """
    Applies the Pauli-X gate to a qubit.
    Args:
        qubit (torch.Tensor): the qubit to apply the gate to
    Returns:
        torch.Tensor: the qubit after the gate is applied
    """
    x_tensor = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
    return x_tensor @ qubit

def Y(qubit):
    """
    Applies the Pauli-Y gate to a qubit.
    Args:
        qubit (torch.Tensor): the qubit to apply the gate to
    Returns:
        torch.Tensor: the qubit after the gate is applied
    """
    y_tensor = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64)
    return y_tensor @ qubit

def Z(qubit):
    """
    Applies the Pauli-Z gate to a qubit.
    Args:
        qubit (torch.Tensor): the qubit to apply the gate to
    Returns:
        torch.Tensor: the qubit after the gate is applied
    """ 
    z_tensor = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)
    return z_tensor @ qubit

def H(qubit):
    """
    Applies the Hadamard gate to a single qubit.

    Args:
        qubit (torch.Tensor): The input qubit state as a tensor.

    Returns:
        torch.Tensor: The resulting qubit state after applying the Hadamard gate.
    """
    h_tensor = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64) / np.sqrt(2)
    return h_tensor @ qubit

def S(qubit):
    """
    Applies the S gate (phase gate) to a single qubit.

    Args:
        qubit (torch.Tensor): The input qubit state as a tensor.

    Returns:
        torch.Tensor: The resulting qubit state after applying the S gate.
    """
    s_tensor = torch.tensor([[1, 0], [0, 1j]], dtype=torch.complex64)
    return s_tensor @ qubit

def T(qubit):
    """
    Applies the T gate (π/8 gate) to a single qubit.

    Args:
        qubit (torch.Tensor): The input qubit state as a tensor.

    Returns:
        torch.Tensor: The resulting qubit state after applying the T gate.
    """
    t_tensor = torch.tensor([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=torch.complex64)
    return t_tensor @ qubit

# two qubit gates
def CNOT(qubit1, qubit2):
    """
    Applies the controlled-NOT (CNOT) gate to two qubits.

    Args:
        qubit1 (torch.Tensor): The input state of the first qubit as a tensor.
        qubit2 (torch.Tensor): The input state of the second qubit as a tensor.

    Returns:
        torch.Tensor: The resulting state after applying the CNOT gate.
    """
    cnot_tensor = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0],
                                [0, 0, 0, 1], [0, 0, 1, 0]], dtype=torch.complex64)
    return cnot_tensor @ torch.kron(qubit1, qubit2)

def BellStateGate(qubit1, qubit2):
    """
    Applies the Bell State gate to two qubits.

    Args:
        qubit1 (torch.Tensor): The input state of the first qubit as a tensor.
        qubit2 (torch.Tensor): The input state of the second qubit as a tensor.

    Returns:
        torch.Tensor: The resulting state after applying the Bell State gate.
    """
    qubit1 = H(qubit1)
    return CNOT(qubit1, qubit2)