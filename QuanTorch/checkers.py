import torch
from .operations import *


# Checkers
def check_orthogonal(vectors):
    """
    Check if the vectors are orthogonal (inner product is zero)
    """
    return all(
        [inner_product(i, j) == 0 for i in vectors for j in vectors if i is not j]
    )


def check_normalized(vector):
    """
    Check if the vector is normalized (inner product is one)
    """
    vectors = [vector, vector]
    return all(
        [
            torch.isclose(inner_product(i, i).real.float(), torch.tensor(1.0))
            for i in vectors
        ]
    )


def check_basis(vectors, normalize=False, verbose=False):
    """
    Check if the vectors are orthogonal and normalized (orthonormal basis)
    """
    # Check if all vectors are orthogonal (inner product is zero)
    orthogonal = check_orthogonal(vectors)

    # Check if all vectors are normalized (inner product is one)
    normalized = True
    for vector in vectors:
        normalized = normalized and check_normalized(vector)

    # initial print
    if verbose:
        print(
            f"Vectors are orthogonal: {orthogonal} | Vectors are normalized: {normalized}"
        )

    # if the vectors are not normalized, normalize them
    if not normalized and normalize and orthogonal:
        # normalize vectors (divide by the norm)
        vectors = [i / torch.sqrt(inner_product(i, i)) for i in vectors]
        if verbose:
            print("Vectors are orthogonal but not normalized. Normalizing...")
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
    coeffs = [torch.tensor(coeff, dtype=torch.complex64) for coeff in coeffs]
    sum = (
        torch.tensor([torch.conj(coeff) * coeff for coeff in coeffs]).real.sum().item()
    )
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


# TODO
def check_unitary(U, verbose=False):
    """
    Check if the matrx is unitary (U * U^dagger = I)
    """
    # check if the matrx is Unitary
    if not torch.allclose(U @ U.T.conj(), torch.eye(U.shape[0], dtype=torch.complex64)):
        if verbose:
            print("Matrix is not Unitary")
        return False
    if verbose:
        print("Unitary matrix")
    return True


def check_valid_density_matrix(P, verbose=False):
    """
    Check if the density matrix is valid
    1) Hermition
    2) trace of P is 1
    3) positive operator
    """
    # check if the matrix is Hermitian
    if not torch.allclose(P, P.T.conj()):
        if verbose:
            print("Matrix is not Hermitian")
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
