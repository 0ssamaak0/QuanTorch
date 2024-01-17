from .helpers import *
import torch
from math import sqrt
from .states import qstate
import QuanTorch.basis as qbasis
from .operations import *


# single qubit gates
def X(qubit):
    """
    Applies the Pauli-X gate to a qubit.
    Args:
        qubit (qstate): the qubit to apply the gate to
    Returns:
        qstate: the qubit after the gate is applied
    """
    x_tensor = qstate([[0, 1], [1, 0]])
    return x_tensor @ qubit


def Y(qubit):
    """
    Applies the Pauli-Y gate to a qubit.
    Args:
        qubit (qstate): the qubit to apply the gate to
    Returns:
        qstate: the qubit after the gate is applied
    """
    y_tensor = qstate([[0, -1j], [1j, 0]])
    return y_tensor @ qubit


def Z(qubit):
    """
    Applies the Pauli-Z gate to a qubit.
    Args:
        qubit (qstate): the qubit to apply the gate to
    Returns:
        qstate: the qubit after the gate is applied
    """
    z_tensor = qstate([[1, 0], [0, -1]])
    return z_tensor @ qubit


def H(qubit):
    """
    Applies the Hadamard gate to a single qubit.

    Args:
        qubit (qstate): The input qubit state as a tensor.

    Returns:
        qstate: The resulting qubit state after applying the Hadamard gate.
    """
    h_tensor = qstate([[1, 1], [1, -1]]) / sqrt(2)
    return h_tensor @ qubit


def S(qubit):
    """
    Applies the S gate (phase gate) to a single qubit.

    Args:
        qubit (qstate): The input qubit state as a tensor.

    Returns:
        qstate: The resulting qubit state after applying the S gate.
    """
    s_tensor = qstate([[1, 0], [0, 1j]])
    return s_tensor @ qubit


def T(qubit):
    """
    Applies the T gate (π/8 gate) to a single qubit.

    Args:
        qubit (qstate): The input qubit state as a tensor.

    Returns:
        qstate: The resulting qubit state after applying the T gate.
    """
    t_tensor = qstate([[1, 0], [0, torch.exp(1j * torch.pi / 4)]])
    return t_tensor @ qubit


# two qubit gates
def CNOT(ctrl, target):
    """
    Applies the controlled-NOT (CNOT) gate to two qubits.

    Args:
        ctrl (qstate): The input state of the first qubit (ctrl) as a tensor.
        target (qstate): The input state of the second qubit (target) as a tensor.

    Returns:
        qstate: The resulting state after applying the CNOT gate.
    """
    cnot_tensor = qstate([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    return cnot_tensor @ tensor_product(ctrl, target)


def BellStateGate(ctrl, target):
    """
    Applies the Bell State gate to two qubits.

    Args:
        ctrl (qstate): The input state of the first qubit (ctrl) as a tensor.
        target (qstate): The input state of the second qubit (target) as a tensor.

    Returns:
        qstate: The resulting state after applying the Bell State gate.
    """
    ctrl = H(ctrl)
    return CNOT(ctrl, target)


def Ctrl_Hadamard(ctrl, target):
    """
    Applies the controlled-Hadamard gate to two qubits.

    Args:
        ctrl (qstate): The input state of the first qubit (ctrl) as a tensor.
        target (qstate): The input state of the second qubit (target) as a tensor.

    Returns:
        qstate: The resulting state after applying the controlled-Hadamard gate.
    """
    chadamard_tensor = qstate(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, sqrt(2) / 2, sqrt(2) / 2],
            [0, 0, sqrt(2) / 2, -sqrt(2) / 2],
        ]
    )
    return chadamard_tensor @ tensor_product(ctrl, target)


def swap_gate(a, b, format="tproduct"):
    """
    Applies the swap gate to two qubits.

    Args:
        a (qstate): The input state of the first qubit as a tensor.
        b (qstate): The input state of the second qubit as a tensor.
        format (str): The format of the input states. (default: "tproduct") (options: "tproduct", "ketbra")
    Returns:
        qstate: The resulting state after applying the swap gate.
    """
    if format == "tproduct":
        swap_tensor = qstate(
            [
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ]
        )
        return swap_tensor @ tensor_product(a, b)
    else:
        return (b, a)


def Ctrl_Swap(ctrl, a, b, format="tproduct"):
    """
    Applies the controlled-swap gate to three qubits.

    Args:
        ctrl (qstate): The input state of the first qubit as a tensor.
        a (qstate): The input state of the second qubit as a tensor.
        b (qstate): The input state of the third qubit as a tensor.
        format (str): The format of the input states. (default: "tproduct") (options: "tproduct", "ketbra")
    Returns:
        qstate: The resulting state after applying the controlled-swap gate.
    """
    swap_tensor = qstate(
        [
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ]
    )
    if torch.isclose(ctrl, qbasis.zero).all():
        if format == "tproduct":
            return tensor_product(a, b)
        else:
            return (a, b)
    elif torch.isclose(ctrl, qbasis.one).all():
        if format == "tproduct":
            return swap_tensor @ tensor_product(a, b)
        else:
            return (b, a)


def Toffoli(a, b, c):
    """
    Applies the Toffolo gate to three qubits.

    Args:
        a (qstate): The input state of the first qubit as a tensor.
        b (qstate): The input state of the second qubit as a tensor.
        c (qstate): The input state of the third qubit as a tensor.
    Returns:
        qstate: The resulting state after applying the Toffolo gate.
    """
    a = 0 if torch.isclose(a, qbasis.zero).all() else 1
    b = 0 if torch.isclose(b, qbasis.zero).all() else 1
    c = 0 if torch.isclose(c, qbasis.zero).all() else 1

    out = c ^ (a & b)
    out = qbasis.zero if out == 0 else qbasis.one

    return out
