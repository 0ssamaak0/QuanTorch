import torch
from .operations import *
from .checkers import *


# Using the density matrix


def determine_state(P, verbose=False):
    """
    Determine if the state is pure, completely mixed or mixed using the trace of P^2
    """
    # check if it's not a valid density matrix
    if not check_valid_density_matrix(P, verbose):
        return "Invalid density matrix"
    P_square = P @ P
    if verbose:
        print(f"Trace of P^2 is {torch.trace(P_square)}")
    # check if trace of P^2 is 1
    if torch.isclose(torch.trace(P_square).real, torch.tensor(1.0)).item():
        return "State is pure"
    # check if trace of P^2 is = 1 / n (n = dimension of P)
    elif torch.isclose(
        torch.trace(P_square).real, torch.tensor(1.0 / P.shape[0])
    ).item():
        return "State is completely mixed"
    else:
        return "State is mixed"


def find_probability(state, basis, verbose=False):
    """
    Find the probability of the state in the given basis
    """
    inner = inner_product(basis, state)
    # multiply inner product with its conjugate
    prob = torch.conj(inner) * inner
    # return the real part of the probability
    return prob.real.item()


def find_expectation(A, P):
    """
    Find the expectation value of the observable A given the density matrix P
    """
    return torch.trace(A @ P).real
