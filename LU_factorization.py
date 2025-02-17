import numpy as np
from scipy.linalg import lu, solve

# Coefficient matrix (A) and constant matrix (b) for LU Factorization
A_lu = np.array([
    [2, -1, 1],
    [3, 3, 9],
    [3, 3, 5]
], dtype=float)

b_lu = np.array([-1, 0, 4], dtype=float)

print("\n--- Coefficient Matrix (A) ---")
for row in A_lu:
    print("  [", "  ".join(f"{val:6.2f}" for val in row), "]")

print("\n--- Constant Matrix (b) ---")
print("  [", "  ".join(f"{val:6.2f}" for val in b_lu), "]")


# LU Factorization
P, L, U = lu(A_lu)

print("\n--- LU Factorization Results ---")
print("Permutation Matrix (P):")
for row in P:
    print("  [", "  ".join(f"{val:6.2f}" for val in row), "]")
print("Lower Triangular Matrix (L):")
for row in L:
    print("  [", "  ".join(f"{val:6.2f}" for val in row), "]")
print("Upper Triangular Matrix (U):")
for row in U:
    print("  [", "  ".join(f"{val:6.2f}" for val in row), "]")

# Solve using LU factorization
# First, solve Ly = Pb, then solve Ux = y
y = solve(L, np.dot(P, b_lu))
x = solve(U, y)

print("\n--- Solution using LU Factorization ---")
for i, xi in enumerate(x):
    print(f"x{i+1} = {xi:.4f}")
