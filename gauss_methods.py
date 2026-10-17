EPSILON = 1e-12
def print_matrix(A, b=None, title="Matrix"):
    print(f"\n{title}:")
    for i, row in enumerate(A):
        values = "  ".join(f"{value:10.4f}" for value in row)
        if b is not None:
            print(f"[ {values} | {b[i]:10.4f} ]")
        else:
            print(f"[ {values} ]")
def back_substitution(U, b):
    n = len(U)
    x = [0.0] * n

    for i in range(n - 1, -1, -1):
        if abs(U[i][i]) < EPSILON:
            raise ValueError("The system has a zero pivot or no unique solution.")

        total = sum(U[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (b[i] - total) / U[i][i]

    return x
def validate_system(A, b):
    if not A or len(A) != len(b):
        raise ValueError("A and b must have the same non-zero number of rows.")

    n = len(A)
    if any(len(row) != n for row in A):
        raise ValueError("A must be a square matrix.")

    if any(not isinstance(value, (int, float)) for row in A for value in row):
        raise ValueError("Matrix A must contain numeric values.")

    if any(not isinstance(value, (int, float)) for value in b):
        raise ValueError("Vector b must contain numeric values.")

    return [[float(value) for value in row] for row in A], [
        float(value) for value in b
    ]
def gauss_row_permutation(A, b):
    A, b = validate_system(A, b)
    n = len(A)

    print("\n" + "=" * 65)
    print("METHOD 1: GAUSSIAN ELIMINATION WITH ROW PERMUTATION")
    print("=" * 65)
    print_matrix(A, b, "Initial augmented matrix")

    for k in range(n - 1):
        # Find a usable pivot row if the current pivot is zero.
        if abs(A[k][k]) < EPSILON:
            pivot_row = next(
                (i for i in range(k + 1, n) if abs(A[i][k]) >= EPSILON),
                None
            )

            if pivot_row is None:
                raise ValueError("Cannot find a non-zero pivot in this column.")

            A[k], A[pivot_row] = A[pivot_row], A[k]
            b[k], b[pivot_row] = b[pivot_row], b[k]
            print(f"\nSwap row {k + 1} with row {pivot_row + 1}.")

        for i in range(k + 1, n):
            multiplier = A[i][k] / A[k][k]
            print(
                f"m[{i + 1},{k + 1}] = "
                f"{A[i][k]:.4f} / {A[k][k]:.4f} = {multiplier:.4f}"
            )

            for j in range(k, n):
                A[i][j] -= multiplier * A[k][j]

            b[i] -= multiplier * b[k]

        print_matrix(A, b, f"After elimination step {k + 1}")

    x = back_substitution(A, b)
    print("\nSolution:")
    for i, value in enumerate(x, start=1):
        print(f"x{i} = {value:.6f}")

    return x
def gauss_partial_pivoting(A, b):
    A, b = validate_system(A, b)
    n = len(A)

    print("\n" + "=" * 65)
    print("METHOD 2: GAUSSIAN ELIMINATION WITH PARTIAL PIVOTING")
    print("=" * 65)
    print_matrix(A, b, "Initial augmented matrix")

    for k in range(n - 1):
        pivot_row = max(range(k, n), key=lambda i: abs(A[i][k]))

        if abs(A[pivot_row][k]) < EPSILON:
            raise ValueError("The system has no unique solution.")

        if pivot_row != k:
            A[k], A[pivot_row] = A[pivot_row], A[k]
            b[k], b[pivot_row] = b[pivot_row], b[k]
            print(f"\nSwap row {k + 1} with row {pivot_row + 1}.")

        print(f"Selected pivot: {A[k][k]:.6f}")

        for i in range(k + 1, n):
            multiplier = A[i][k] / A[k][k]

            for j in range(k, n):
                A[i][j] -= multiplier * A[k][j]

            b[i] -= multiplier * b[k]

        print_matrix(A, b, f"After elimination step {k + 1}")

    x = back_substitution(A, b)
    print("\nSolution:")
    for i, value in enumerate(x, start=1):
        print(f"x{i} = {value:.6f}")

    return x
def gauss_total_pivoting(A, b):
    A, b = validate_system(A, b)
    n = len(A)
    column_order = list(range(n))

    print("\n" + "=" * 65)
    print("METHOD 3: GAUSSIAN ELIMINATION WITH TOTAL PIVOTING")
    print("=" * 65)
    print_matrix(A, b, "Initial augmented matrix")

    for k in range(n - 1):
        pivot_row, pivot_col = max(
            ((i, j) for i in range(k, n) for j in range(k, n)),
            key=lambda position: abs(A[position[0]][position[1]])
        )

        if abs(A[pivot_row][pivot_col]) < EPSILON:
            raise ValueError("The system has no unique solution.")

        if pivot_row != k:
            A[k], A[pivot_row] = A[pivot_row], A[k]
            b[k], b[pivot_row] = b[pivot_row], b[k]
            print(f"\nSwap row {k + 1} with row {pivot_row + 1}.")

        if pivot_col != k:
            for row in A:
                row[k], row[pivot_col] = row[pivot_col], row[k]

            column_order[k], column_order[pivot_col] = (
                column_order[pivot_col],
                column_order[k]
            )
            print(f"Swap column {k + 1} with column {pivot_col + 1}.")

        print(f"Selected pivot: {A[k][k]:.6f}")

        for i in range(k + 1, n):
            multiplier = A[i][k] / A[k][k]

            for j in range(k, n):
                A[i][j] -= multiplier * A[k][j]

            b[i] -= multiplier * b[k]

        print_matrix(A, b, f"After elimination step {k + 1}")

    z = back_substitution(A, b)

    
    x = [0.0] * n
    for current_position, original_position in enumerate(column_order):
        x[original_position] = z[current_position]

    print("\nSolution in the original variable order:")
    for i, value in enumerate(x, start=1):
        print(f"x{i} = {value:.6f}")

    return x

def main():
    A1 = [
        [0, 2],
        [3, 1]
    ]
    b1 = [4, 7]

    A2 = [
        [0.001, 1],
        [1, 1]
    ]
    b2 = [1, 2]

    A3 = [
        [1, 2],
        [3, 4]
    ]
    b3 = [5, 11]

    gauss_row_permutation(A1, b1)
    gauss_partial_pivoting(A2, b2)
    gauss_total_pivoting(A3, b3)


if __name__ == "__main__":
    main()
