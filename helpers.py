import torch
import numpy as np

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

def check_orthogonal(vectors):
    """
    Check if the vectors are orthogonal (inner product is zero)
    """
    return all([inner_product(i, j) == 0 for i in vectors for j in vectors if i is not j])


def check_normalized(vectors):
    """
    Check if the vectors are normalized (inner product is one)
    """
    return all([np.isclose(inner_product(i, i).real.float(), 1) for i in vectors])


def check_basis(vectors, normalize=False, verbose=True):
    """
    Check if the vectors are orthogonal and normalized (orthonormal basis)
    """ 
    # Check if all vectors are orthogonal (inner product is zero)
    orthogonal = check_orthogonal(vectors)

    # Check if all vectors are normalized (inner product is one)
    normalized = check_normalized(vectors)

    # initial print
    if verbose:
        print(f'Vectors are orthogonal: {orthogonal} | Vectors are normalized: {normalized}')

    # if the vectors are not normalized, normalize them
    if not normalized and normalize and orthogonal:
        # normalize vectors (divide by the norm)
        vectors = [i / torch.sqrt(inner_product(i, i)) for i in vectors]
        if verbose:
            print('Vectors are orthogonal but not normalized. Normalizing...')
            print(f"vectors after normalization:")
            for i in vectors:
                print(i)

    return orthogonal and normalized


def check_qubit(vector, verbose=False):
    """
    Check if the qubit is valid (sum of coeff.conj * coeff is one)
    """
    # 1) Check the basis first?
    # 2) Check sum of coeff.conj * coeff = 1
    coeffs = vector.tolist()
    coeffs = [torch.tensor(coeff, dtype = torch.complex64) for coeff in coeffs]
    sum =  torch.tensor([torch.conj(coeff) * coeff for coeff in coeffs]).real.sum().item()
    if torch.isclose(torch.tensor(sum), torch.tensor(1.0)).item():
        return True
    else:
        if verbose:
            print(f"Sum of coeff.conj * coeff is {sum:0.4} instead of 1.0")
        return False


def positive_operator(P, verbose=False):
    """
    Check if the operator is positive (all eigenvalues are positive)
    """
    # Find the eigenvalues of the operator matrix
    L, _ = torch.linalg.eig(P)
    # convert eigenvalues to a list of real numbers
    L = L.real.tolist()
    # check if all eigenvalues are positive
    result = all([l > 0 for l in L])
    if result:
        if verbose:
            print("Positive Operator")
        return True
    else:
        if verbose:
            print(f"Eigen values are not positive\n{L}")
        return False
    
def check_valid_density_matrix(P, verbose=False):
    """
    Check if the density matrix is valid 
    1) Hermition
    2) trace of P is 1
    3) positive operator
    """
    # check if the matrix is hermition
    if not torch.allclose(P, P.T.conj()):
        if verbose:
            print("Matrix is not hermition")
        return False
    # check if trace of P is 1
    if not torch.isclose(torch.trace(P).real, torch.tensor(1.0)).item():
        if verbose:
            print(f"Trace of P is {torch.trace(P)} instead of 1.0")
        return False
    # check if the operator is positive
    if not positive_operator(P):
        if verbose:
            print("Operator is not positive")
        return False
    if verbose:
        print("Valid density matrix")
    return True


def determine_state(P, verbose=False):
    """
    Determine if the state is pure, completely mixed or mixed using the trace of P^2
    """
    # check if it's not a valid density matrix
    if not check_valid_density_matrix(P, verbose):
        print("Invalid density matrix")
        return False
    P_square = P @ P
    if verbose:
        print(f"Trace of P^2 is {torch.trace(P_square)}")
    # check if trace of P^2 is 1
    if torch.isclose(torch.trace(P_square).real, torch.tensor(1.0)).item():
        print("State is pure")
        return True
    # check if trace of P^2 is = 1 / n (n = dimension of P)
    elif torch.isclose(torch.trace(P_square).real, torch.tensor(1.0 / P.shape[0])).item():
        print("State is completely mixed")
        return True
    else:
        print("State is mixed")
        return True
    
def find_expectation(A, P):
    """
    Find the expectation value of the observable A given the density matrix P
    """
    return torch.trace(A @ P).real