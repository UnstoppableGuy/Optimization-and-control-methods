import sys
import numpy as np

from lab1 import input_matrix, input_vector
# from lab4 import first_and_second_step_simplex_method


def double_simplex(c, b, a_matrix, j_vector):
    """Сreates an optimal unfeasible plan and then converts
    it to a feasible one without violating optimality.

    Args:
        c (np.array): vector of values
        b (np.array): vector of values
        a_matrix (np.array): matrix composed of the coefficients
        of the original system
        j_vector (np.array): vector of values

    Returns:
        any: feasible plan (list) or message (str)
    """
    m, n = a_matrix.shape
    j_vector -= 1
    y = get_initial_y(c, a_matrix, j_vector)
    x_0 = [0 for _ in range(n)]
    while True:
        not_J = np.delete(np.arange(n), j_vector)
        B = np.linalg.inv(a_matrix[:, j_vector])
        kappa = B.dot(b)
        if all(kappa >= 0):
            for j, _kappa in zip(j_vector, kappa):
                x_0[j] = _kappa
            print(str(list(map(lambda _x: round(float(_x), 3), list(x_0)))
                      ).replace('[', '').replace(']', ''), "-  план")
            print(f"План: \t{' '.join(map(str,list(x_0)))}")
            return x_0
        k = np.argmin(kappa)
        delta_y = B[k]
        mu = delta_y.dot(a_matrix)
        sigma = []
        for i in not_J:
            if mu[i] >= 0:
                sigma.append(np.inf)
            else:
                sigma.append((c[i] - a_matrix[:, i].dot(y)) / mu[i])
        sigma_0_ind = not_J[np.argmin(sigma)]
        sigma_0 = min(sigma)
        if sigma_0 == np.inf:
            print("Задача не имеет решения, т.к. пусто множество ее\
                допустимых планов.")
            return "Задача не имеет решения"

        y += sigma_0 * delta_y
        j_vector[k] = sigma_0_ind


def get_initial_y(c, a_matrix, j_vector):
    return (c[j_vector]).dot(np.linalg.inv(a_matrix[:, j_vector]))


def test1():
    """test case 1

    Returns:
        matrix: np.array
        vector b: np.array
        vector c: np.array
        vector j: np.array
    """
    A = np.array([
        [-2, -1, -4, 1, 0],
        [-2, -2, -2, 0, 1]
    ])
    b = np.array([-1, -1.5])
    c = np.array([-4, -3, -7, 0, 0])
    J = np.array([4, 5])
    double_simplex(c=c, b=b, a_matrix=A, j_vector=J)
    return A, b, c, J


def test2():
    """test case 2

    Returns:
        matrix: np.array
        vector b: np.array
        vector c: np.array
        vector j: np.array
    """
    A = np.array([
        [-2, -1, 1, -7, 0, 0, 0, 2],
        [4, 2, 1, 0, 1, 5, -1, -5],
        [1, 1, 0, -1, 0, 3, -1, 1]
    ])
    b = np.array([-2, 4, 3])
    c = np.array([2, 2, 1, -10, 1, 4, -2, -3])
    J = np.array([2, 5, 7])
    double_simplex(c=c, b=b, a_matrix=A, j_vector=J)
    return A, b, c, J


def test3():
    """test case 3

    Returns:
        matrix: np.array
        vector b: np.array
        vector c: np.array
        vector j: np.array
    """
    A = np.array([
        [-2, -1, 1, -7, 0, 0, 0, 2],
        [-4, 2, 1, 0, 1, 5, -1, 5],
        [1, 1, 0, 1, 4, 3, 1, 1]
    ])
    b = np.array([-2, 8, -2])
    c = np.array([12, -2, -6, 20, -18, -5, -7, -20])
    J = np.array([2, 4, 6])
    double_simplex(c=c, b=b, a_matrix=A, j_vector=J)
    return A, b, c, J


def test4():
    """test case 4

    Returns:
        matrix: np.array
        vector b: np.array
        vector c: np.array
        vector j: np.array
    """
    A = np.array([
        [-2, -1, 10, -7, 1, 0, 0, 2],
        [-4, 2, 3, 0, 5, 1, -1, 0],
        [1, 1, 0, 1, -4, 3, -1, 1]
    ])
    b = np.array([-2, -5, 2])
    c = np.array([10, -2, -38, 16, -9, -9, -5, -7])
    J = np.array([2, 8, 5])
    double_simplex(c=c, b=b, a_matrix=A, j_vector=J)
    return A, b, c, J


def simplex():
    if 'test' in sys.argv:
        test1()
        test2()
        test3()
        test4()
    else:
        m, n = map(int, input('Введите количество строк и столбцов').split())
        matrix_a = input_matrix(m)
        vector_b = input_vector(m)
        vector_c = input_vector(m)
        vector_j = input_vector(m)
        double_simplex(c=vector_c, a_matrix=matrix_a,
                       b=vector_b, j_vector=vector_j)

        # vector_x = first_and_second_step_simplex_method(
        #     matrix_a, vector_b, vector_c)

        # if vector_x is None:
        #     print("Unbounded")
        # elif len(vector_x) == 0:
        #     print("No solution")
        # else:
        #     print(f"Bounded\n{' '.join(map(str, vector_x))}")


if __name__ == "__main__":
    simplex()
