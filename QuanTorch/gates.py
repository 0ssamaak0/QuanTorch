from .helpers import *
import torch
from math import sqrt
from .states import qstate
import QuanTorch.basis as qbasis
from .operations import *


# single qubit gates
def X(qubit, gate_matrix=False):
    """
    Applies the Pauli-X gate to a qubit.
    Args:
        qubit (qstate): the qubit to apply the gate to
        gate_matrix (bool): if True, returns the gate matrix instead of the qubit after the gate is applied
    Returns:
        qstate: the qubit after the gate is applied
    """
    x_tensor = qstate([[0, 1], [1, 0]])

    if gate_matrix:
        return x_tensor

    return x_tensor @ qubit


def Y(qubit, gate_matrix=False):
    """
    Applies the Pauli-Y gate to a qubit.
    Args:
        qubit (qstate): the qubit to apply the gate to
        gate_matrix (bool): if True, returns the gate matrix instead of the qubit after the gate is applied
    Returns:
        qstate: the qubit after the gate is applied
    """
    y_tensor = qstate([[0, -1j], [1j, 0]])
    if gate_matrix:
        return y_tensor
    return y_tensor @ qubit


def Z(qubit, gate_matrix=False):
    """
    Applies the Pauli-Z gate to a qubit.
    Args:
        qubit (qstate): the qubit to apply the gate to
        gate_matrix (bool): if True, returns the gate matrix instead of the qubit after the gate is applied
    Returns:
        qstate: the qubit after the gate is applied
    """
    z_tensor = qstate([[1, 0], [0, -1]])
    if gate_matrix:
        return z_tensor
    return z_tensor @ qubit


def H(qubit, gate_matrix=False):
    """
    Applies the Hadamard gate to a single qubit.

    Args:
        qubit (qstate): The input qubit state as a tensor.
        gate_matrix (bool): if True, returns the gate matrix instead of the qubit after the gate is applied

    Returns:
        qstate: The resulting qubit state after applying the Hadamard gate.
    """
    h_tensor = qstate([[1, 1], [1, -1]]) / sqrt(2)
    if gate_matrix:
        return h_tensor
    return h_tensor @ qubit


def S(qubit, gate_matrix=False):
    """
    Applies the S gate (phase gate) to a single qubit.

    Args:
        qubit (qstate): The input qubit state as a tensor.
        gate_matrix (bool): if True, returns the gate matrix instead of the qubit after the gate is applied

    Returns:
        qstate: The resulting qubit state after applying the S gate.
    """
    s_tensor = qstate([[1, 0], [0, 1j]])
    if gate_matrix:
        return s_tensor
    return s_tensor @ qubit


def T(qubit, gate_matrix=False):
    """
    Applies the T gate (π/8 gate) to a single qubit.

    Args:
        qubit (qstate): The input qubit state as a tensor.
        gate_matrix (bool): if True, returns the gate matrix instead of the qubit after the gate is applied

    Returns:
        qstate: The resulting qubit state after applying the T gate.
    """
    t_tensor = qstate([[1, 0], [0, torch.exp(1j * torch.pi / 4)]])
    if gate_matrix:
        return t_tensor
    return t_tensor @ qubit


# two qubit gates
def CNOT(ctrl, target, gate_matrix=False):
    """
    Applies the controlled-NOT (CNOT) gate to two qubits.

    Args:
        ctrl (qstate): The input state of the first qubit (ctrl) as a tensor.
        target (qstate): The input state of the second qubit (target) as a tensor.
        gate_matrix (bool): if True, returns the gate matrix instead of the qubit after the gate is applied

    Returns:
        qstate: The resulting state after applying the CNOT gate.
    """
    cnot_tensor = qstate([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    if gate_matrix:
        return cnot_tensor
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


def Ctrl_Hadamard(ctrl, target, gate_matrix=False):
    """
    Applies the controlled-Hadamard gate to two qubits.

    Args:
        ctrl (qstate): The input state of the first qubit (ctrl) as a tensor.
        target (qstate): The input state of the second qubit (target) as a tensor.
        gate_matrix (bool): if True, returns the gate matrix instead of the qubit after the gate is applied

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
    if gate_matrix:
        return chadamard_tensor
    return chadamard_tensor @ tensor_product(ctrl, target)


def swap_gate(a, b, format="tproduct", gate_matrix=False):
    """
    Applies the swap gate to two qubits.

    Args:
        a (qstate): The input state of the first qubit as a tensor.
        b (qstate): The input state of the second qubit as a tensor.
        format (str): The format of the input states. (default: "tproduct") (options: "tproduct", "ketbra")
        gate_matrix (bool): if True, returns the gate matrix instead of the qubit after the gate is applied
    Returns:
        qstate: The resulting state after applying the swap gate.
    """
    swap_tensor = qstate(
        [
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ]
    )
    if gate_matrix:
        return swap_tensor
    if format == "tproduct":
        return swap_tensor @ tensor_product(a, b)
    else:
        return (b, a)


def Ctrl_Swap(ctrl, a, b, format="tproduct", gate_matrix=False):
    """
    Applies the controlled-swap gate to three qubits.

    Args:
        ctrl (qstate): The input state of the first qubit as a tensor.
        a (qstate): The input state of the second qubit as a tensor.
        b (qstate): The input state of the third qubit as a tensor.
        format (str): The format of the input states. (default: "tproduct") (options: "tproduct", "ketbra")
        gate_matrix (bool): if True, returns the gate matrix instead of the qubit after the gate is applied
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
    if gate_matrix:
        return swap_tensor
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
