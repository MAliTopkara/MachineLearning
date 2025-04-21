import numpy as np
import time

# ----------------------------
# Manuel Özdeğer-Özvektör Hesabı
# ----------------------------
def characteristic_polynomial_3x3(matrix):
    trace = np.trace(matrix)
    minor_sum = (
        matrix[0, 0]*matrix[1, 1] + matrix[0, 0]*matrix[2, 2] + matrix[1, 1]*matrix[2, 2]
        - matrix[0, 1]*matrix[1, 0] - matrix[0, 2]*matrix[2, 0] - matrix[1, 2]*matrix[2, 1]
    )
    det = (matrix[0, 0]*matrix[1, 1]*matrix[2, 2] +
           matrix[0, 1]*matrix[1, 2]*matrix[2, 0] +
           matrix[0, 2]*matrix[1, 0]*matrix[2, 1] -
           matrix[0, 2]*matrix[1, 1]*matrix[2, 0] -
           matrix[0, 0]*matrix[1, 2]*matrix[2, 1] -
           matrix[0, 1]*matrix[1, 0]*matrix[2, 2])
    return [-1, trace, -minor_sum, det]

def rref(matrix):
    A = matrix.astype(float).copy()
    rows, cols = A.shape
    pivot_row = 0
    for col in range(cols):
        if pivot_row >= rows:
            break
        pivot = None
        for row in range(pivot_row, rows):
            if abs(A[row, col]) > 1e-10:
                pivot = row
                break
        if pivot is None:
            continue
        A[[pivot_row, pivot]] = A[[pivot, pivot_row]]
        A[pivot_row] = A[pivot_row] / A[pivot_row, col]
        for r in range(rows):
            if r != pivot_row and abs(A[r, col]) > 1e-10:
                A[r] -= A[r, col] * A[pivot_row]
        pivot_row += 1
    return A

def nullspace_rref(A):
    A_rref = rref(A)
    m, n = A_rref.shape
    pivot_cols = []
    for i in range(m):
        for j in range(n):
            if abs(A_rref[i, j]) > 1e-10:
                pivot_cols.append(j)
                break
    free_cols = [j for j in range(n) if j not in pivot_cols]

    null_vectors = []
    for free in free_cols:
        vec = np.zeros(n)
        vec[free] = 1
        for i in range(len(pivot_cols)):
            pivot_col = pivot_cols[i]
            vec[pivot_col] = -A_rref[i, free]
        norm = sum(x**2 for x in vec)**0.5
        if norm > 1e-10:
            vec = vec / norm
        null_vectors.append(vec)
    return np.array(null_vectors).T

def eig_manual_no_lib(matrix):
    start = time.time()
    n = matrix.shape[0]
    if n != 3:
        raise ValueError("Sadece 3x3 matrisler desteklenmektedir.")

    coeffs = characteristic_polynomial_3x3(matrix)
    lambdas = np.roots(coeffs)

    eigenvectors = []
    for lam in lambdas:
        A_lamI = matrix - lam * np.eye(n)
        null_vecs = nullspace_rref(A_lamI)
        eigenvectors.append(null_vecs[:, 0])  # ilk özvektörü al

    end = time.time()
    return lambdas, np.column_stack(eigenvectors), end - start

# ----------------------------
# Matris Tanımı
# ----------------------------
A = np.array([[6, 1, -1],
              [0, 7, 0],
              [3, -1, 2]], dtype=float)

# ----------------------------
# Manuel Yöntem
# ----------------------------
eigenvalues_manual, eigenvectors_manual, duration_manual = eig_manual_no_lib(A)

print("==== Manuel Hesaplama ====")
print("Özdeğerler:\n", eigenvalues_manual)
print("\nÖzvektörler:\n", eigenvectors_manual)
print("\nSüre: {:.6f} saniye".format(duration_manual))

# ----------------------------
# NumPy ile Yöntem
# ----------------------------
start_np = time.time()
eigvals_np, eigvecs_np = np.linalg.eig(A)
end_np = time.time()
duration_np = end_np - start_np

print("\n==== NumPy eig Fonksiyonu ile ====")
print("Özdeğerler:\n", eigvals_np)
print("\nÖzvektörler:\n", eigvecs_np)
print("\nSüre: {:.6f} saniye".format(duration_np))

# ----------------------------
# Karşılaştırma
# ----------------------------
print("\n==== Karşılaştırma ====")
print("Özdeğerler yaklaşık eşit mi?:", np.allclose(np.sort(eigenvalues_manual), np.sort(eigvals_np)))
print("Özvektörler yaklaşık eşit mi? (mutlak değer kontrolü):",
      np.allclose(np.abs(eigenvectors_manual), np.abs(eigvecs_np), atol=1e-2))






