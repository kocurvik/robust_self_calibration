import numpy as np
from scipy.linalg import lu

from utils.geometry import skew

def gauss_elimination(A):
    row, col = A.shape
    if row == 0 or col == 0:
        return A
    for i in range(min(row, col)):
        max_element = abs(A[i, i])
        max_row = i
        for k in range(i+1, row):
            if abs(A[k, i]) > max_element:
                max_element = abs(A[k, i])
                max_row = k
        A[[i, max_row]] = A[[max_row, i]]
        for k in range(i+1, row):
            factor = A[k, i]/A[i, i]
            A[k, :] -= factor * A[i, :]
    return A


def row_echelon(A):
    """ Return Row Echelon Form of matrix A """

    # if matrix A has no columns or rows,
    # it is already in REF, so we return itself
    r, c = A.shape
    if r == 0 or c == 0:
        return A

    # we search for non-zero element in the first column
    for i in range(len(A)):
        if A[i,0] != 0:
            break
    else:
        # if all elements in the first column is zero,
        # we perform REF on matrix from second column
        B = row_echelon(A[:,1:])
        # and then add the first zero-column back
        return np.hstack([A[:,:1], B])

    # if non-zero element happens not in the first row,
    # we switch rows
    if i > 0:
        ith_row = A[i].copy()
        A[i] = A[0]
        A[0] = ith_row

    # we divide first row by first element in it
    A[0] = A[0] / A[0,0]
    # we subtract all subsequent rows with first row (it has 1 now as first element)
    # multiplied by the corresponding element in the first column
    A[1:] -= A[0] * A[1:,0:1]

    # we perform REF on matrix from second row, from second column
    B = row_echelon(A[1:,1:])

    # we add first row and first (zero) column, and return
    return np.vstack([A[:1], np.hstack([A[1:,:1], B]) ])

def melanitis(F):
    u, s, vt = np.linalg.svd(F)

    a = u[:, 2]

    P = np.column_stack([skew(a) @ F, a])

    p11, p12, p13, p14 = P[0, :]
    p21, p22, p23, p24 = P[1, :]
    p31, p32, p33, p34 = P[2, :]

    A = np.array([
        [p21 ** 2 + p22 ** 2,   -1,     p24 ** 2,    -2 * p23 * p24,         -2 * p21 * p24,         - 2 * p22 * p24,           -p23**2],
        [p21 * p31 + p22 * p32,  0,     p24 * p34,   -p23 * p34 - p24 * p33, -p21 * p34 - p24 * p31, -p22 * p34 - p24 * p32,    -p23 * p33],
        [p11 * p31 + p12 * p32,  0,     p14 * p34,   -p13 * p34 - p14 * p33, -p11 * p34 - p14 * p31, -p12 * p34 - p14 * p32,    -p13 * p33],
        [p11 ** 2 + p12 ** 2,   -1,     p14 ** 2,    -2 * p13 * p14,         -2 * p11 * p14,         -2 * p12 * p14,            -p13**2],
        [p11 * p21 + p12 * p22,  0,     p14 * p24,   -p13 * p24 - p14 * p23, -p11 * p24 - p14 * p21, -p12 * p24 - p14 * p22,    -p13 * p23]
    ])

    # _, _, u = lu(A)

    # b = u[:, -1]

    A_dag = np.linalg.pinv(A[:, :6])

    b = A_dag @ A[:, -1]

    b = np.linalg.solve(A[:, :5], A[:, -1])

    # AA = row_echelon(A)
    # b = AA[:, -1]

    f1 = np.sqrt(b[0])
    f2 = np.sqrt(b[1])

    return f1, f2