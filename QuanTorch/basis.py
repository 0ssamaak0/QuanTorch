from math import sqrt
from .states import qstate

# Define the basis vectors | 0 > and | 1 >, and the Hadamard | + > and | - > states
zero = qstate([[1], [0]])
one = qstate([[0], [1]])
plus = qstate([[1], [1]]) / sqrt(2)
minus = qstate([[1], [-1]]) / sqrt(2)
