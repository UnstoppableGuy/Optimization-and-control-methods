import numpy as np
import math


def input_matrix(size):
    """enter matrix

    Args:
        size (int): size for matrix

    Returns:
        np.array: matrix with args
    """
    matrix = np.array([[float(j) for j in input("Строка матрицы: ").split()]
                       for i in range(size)])
    return matrix


def input_vector():
    """enter vector

    Returns:
        list: vector with args
    """
    return list(map(int, input('Элементы вектора: ').split()))


def get_inverted(matrix_b, vector_x, position):
    """_summary_

    Args:
        matrix_b (np.array): source matrix
        vector_x (np.array): plan
        position (np.array): index

    Returns:
        np.array: inverted matrix_
    """
    vector_l = matrix_b.dot(vector_x.T)
    if vector_l[position] == 0:
        return None

    vector_l_cover = vector_l[position]
    vector_l[position] = -1
    vector_l *= -1 / vector_l_cover

    matrix_b_new = np.eye(len(matrix_b), dtype=float)
    matrix_b_new[:, position] = vector_l

    return matrix_b_new.dot(matrix_b)


def main_stage_simplex_method(m, n, matrix_a, vector_b, vector_c, vector_x,
                              vector_jb):
    """main stage simplex

    Args:
        m (int): count of rows
        n (int): count of colums
        matrix_a (np.array): matrix with values
        vector_b (list): vector b
        vector_c (list): vector c
        vector_x (list): vector x
        vector_jb (list): vector jb

    Returns:
        object: None or list
    """
    if m == n:
        return vector_x
    matrix_ab = matrix_a[:, vector_jb]
    matrix_b = np.linalg.inv(matrix_ab)

    while True:
        vector_jb_n = [i for i in range(n) if i not in vector_jb]

        delta = vector_c[vector_jb].dot(matrix_b).dot(
            matrix_a[:, vector_jb_n]) - vector_c[vector_jb_n]

        checker = -1
        for i, el in enumerate(delta):
            if el < 0:
                checker = i
                break
        if checker == -1:
            return vector_x

        j0 = vector_jb_n[checker]

        vector_z = matrix_b.dot(matrix_a[:, j0])
        if all([i <= 0 for i in vector_z]):
            return None

        theta = [vector_x[vector_jb[i]] / vector_z[i]
                 if vector_z[i] > 0 else math.inf for i in range(m)]

        theta_0 = min(theta)
        s = theta.index(theta_0)
        vector_jb[s] = j0

        matrix_b = get_inverted(matrix_b, matrix_a[:, j0], s)

        if matrix_b is None:
            return None

        vector_x_new = np.zeros(n, dtype=float)
        vector_x_new[vector_jb] = vector_x[vector_jb] - theta_0 * vector_z
        vector_x_new[j0] = theta_0
        vector_x = vector_x_new


def first_step_simplex_method(matrix_a, vector_b, m, n):
    for i in range(m):
        if vector_b[i] < 0:
            vector_b[i] *= -1
            matrix_a[i] *= -1

    vector_jb = [i for i in range(n, n + m)]
    zeros = [0. for i in range(n)]
    ones = [1. for i in range(m)]

    matrix = np.concatenate((matrix_a, np.eye(m)), axis=1)

    vector_c = np.array(zeros+ones)
    vector_x_start = np.array(zeros+vector_b.copy())

    vector_x = main_stage_simplex_method(
        m, n + m, matrix, vector_b, -vector_c, vector_x_start, vector_jb)

    if vector_x is None:
        return None

    vector_x_0 = vector_x[:n]
    vector_x_u = vector_x[n:]

    if any(vector_x_0 < 0) or any(vector_x_u != 0):
        return ()

    return vector_x_0


def test1():
    """test case 1

    Returns:
        np.array: matrix
        np.array: vector
    """
    A = np.array([
        [1, 1, 1],
        [2, 2, 2],
        [3, 3, 3]])
    b = [0, 0, 0]

    return A, b


def test2():
    """test case 2

    Returns:
        np.array: matrix
        np.array: vector
    """
    A = np.array([
        [0.0, 1.0, 4.0, 1.0, 0.0, -8.0, 1.0, 5.0],
        [0.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 2.0, -1.0, 0.0, -1.0, 3.0, -1.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 0.0, 3.0, 1.0, 1.0]
    ])
    b = [36.0, -11.0, 10.0, 20.0]

    return A, b


def simplex():
    """simplex method

    Returns:
        str: plan for this args
    """
    import sys
    if 'test' in sys.argv:
        matrix_a, vector_b = test1()
        m, n = matrix_a.shape
    else:
        m, n = map(int, input().split())
        matrix_a = input_matrix(m)
        vector_b, vector_c = input_vector(), input_vector()

    vector_x = first_step_simplex_method(matrix_a, vector_b, m, n)

    if vector_x is None:
        return "Unbounded"
    elif len(vector_x) == 0:
        return "No solution"
    else:
        return f"Bounded\n{' '.join(map(str, vector_x))}"


if __name__ == "__main__":
    print(simplex())
