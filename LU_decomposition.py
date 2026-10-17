EPSILON = 1e-12


def print_matrix(M, title):
    print(f"\n{title}:")
    for row in M:
        print("[ " + "  ".join(f"{value:10.4f}" for value in row) + " ]")


def print_vector(v, title):
    print(f"\n{title}:")
    for i, value in enumerate(v, start=1):
        print(f"  {title.split()[0]}{i} = {value:.6f}")


def validate_system(A, b):
    if not A or len(A) != len(b):
        raise ValueError("A and b must have the same non-zero number of rows.")

    n = len(A)
    if any(len(row) != n for row in A):
        raise ValueError("A must be a square matrix.")

    return (
        [[float(value) for value in row] for row in A],
        [float(value) for value in b]
    )


def lu_decomposition(A):
    if not A or any(len(row) != len(A) for row in A):
        raise ValueError("A must be a non-empty square matrix.")

    n = len(A)
    U = [[float(value) for value in row] for row in A]
    L = [[0.0] * n for _ in range(n)]
    P = [[float(i == j) for j in range(n)] for i in range(n)]

    for i in range(n):
        L[i][i] = 1.0

    print_matrix(U, "Initial matrix A")

    for k in range(n):
        # Find the largest absolute pivot in column k.
        pivot_row = max(range(k, n), key=lambda i: abs(U[i][k]))

        if abs(U[pivot_row][k]) < EPSILON:
            raise ValueError("LU decomposition failed: matrix is singular.")

        if pivot_row != k:
            U[k], U[pivot_row] = U[pivot_row], U[k]
            P[k], P[pivot_row] = P[pivot_row], P[k]

            # Swap only the previously calculated entries of L.
            for j in range(k):
                L[k][j], L[pivot_row][j] = L[pivot_row][j], L[k][j]

            print(f"\nPivoting: swap row {k + 1} with row {pivot_row + 1}.")

        for i in range(k + 1, n):
            multiplier = U[i][k] / U[k][k]
            L[i][k] = multiplier

            for j in range(k, n):
                U[i][j] -= multiplier * U[k][j]

            U[i][k] = 0.0

        print_matrix(L, f"L after step {k + 1}")
        print_matrix(U, f"U after step {k + 1}")

    return P, L, U


def forward_substitution(L, b):
    n = len(L)
    y = [0.0] * n

    for i in range(n):
        if abs(L[i][i]) < EPSILON:
            raise ValueError("Zero diagonal entry in L.")

        total = sum(L[i][j] * y[j] for j in range(i))
        y[i] = (b[i] - total) / L[i][i]

    return y


def back_substitution(U, y):
    n = len(U)
    x = [0.0] * n

    for i in range(n - 1, -1, -1):
        if abs(U[i][i]) < EPSILON:
            raise ValueError("Zero diagonal entry in U.")

        total = sum(U[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (y[i] - total) / U[i][i]

    return x


def multiply_matrix_vector(A, v):
    return [
        sum(A[i][j] * v[j] for j in range(len(v)))
        for i in range(len(A))
    ]


def multiply_matrices(A, B):
    rows = len(A)
    cols = len(B[0])
    inner = len(B)

    return [
        [
            sum(A[i][k] * B[k][j] for k in range(inner))
            for j in range(cols)
        ]
        for i in range(rows)
    ]


def solve_lu(A, b):
    A, b = validate_system(A, b)

    print("\n" + "=" * 65)
    print("LU DECOMPOSITION WITH PARTIAL PIVOTING")
    print("=" * 65)

    P, L, U = lu_decomposition(A)

    print_matrix(P, "Permutation matrix P")
    print_matrix(L, "Lower-triangular matrix L")
    print_matrix(U, "Upper-triangular matrix U")

    Pb = multiply_matrix_vector(P, b)
    print_vector(Pb, "Pb")

    y = forward_substitution(L, Pb)
    print_vector(y, "y")

    x = back_substitution(U, y)
    print_vector(x, "x")

    PA = multiply_matrices(P, A)
    LU = multiply_matrices(L, U)

    factorization_error = max(
        abs(PA[i][j] - LU[i][j])
        for i in range(len(A))
        for j in range(len(A))
    )

    Ax = multiply_matrix_vector(A, x)
    residual = max(abs(Ax[i] - b[i]) for i in range(len(b)))

    print(f"\nMaximum factorization error ||PA - LU||_max = "
          f"{factorization_error:.3e}")
    print(f"Maximum residual ||Ax - b||_max = {residual:.3e}")

    if factorization_error < 1e-9 and residual < 1e-9:
        print("\nVerification successful: the results are correct.")
    else:
        print("\nWarning: check the numerical results.")

    return x


def main():

    A = [
        [2, 1, 1],
        [4, 3, 3],
        [8, 7, 9]
    ]

    b = [4, 10, 24]

    solution = solve_lu(A, b)

    print("\nFinal solution:")
    for i, value in enumerate(solution, start=1):
        print(f"x{i} = {value:.6f}")


if __name__ == "__main__":
    main()
