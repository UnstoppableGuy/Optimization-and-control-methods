import sys
import numpy as np
import json


def input_matrix(m):
    matrix = []
    for el in range(m):
        matrix.append(list(map(int, input().split())))
    return np.array(matrix)


def input_vector():
    return list(map(int, input().split()))


def input_float_vector():
    return list(map(float, input().split()))


def create_matrix_ab(matrix_a, vector_jb):
    size = len(vector_jb)
    matrix_ab = np.zeros((size, size))

    for index, el in enumerate(vector_jb):
        matrix_ab[:, index] = matrix_a[:, el-1]

    return matrix_ab


def create_vector_cb(vector_c, vector_jb):
    vector_cb = np.zeros(len(vector_jb))

    for i, el in enumerate(vector_jb):
        vector_cb[i] = vector_c[el-1]

    return vector_cb


def find_min_delta(vector_delta, all_j, vector_jb):
    vector_jb_H = all_j - set(vector_jb)

    if vector_jb_H == set():
        return (1, 1)

    min_delta = []
    for i in vector_jb_H:
        min_delta.append(vector_delta[i-1])

    delta = min(min_delta)
    return (min_delta.index(delta), delta)


def min_theta(vector_x, vector_z, vector_jb):
    vector_theta = []
    min_index = 0
    min_value = -1
    i = 0
    for z, jb in zip(vector_z, vector_jb):
        if z > 0:
            theta = vector_x[jb-1] / z
            vector_theta.append(theta)
            if theta <= min(vector_theta):
                min_index = i
                min_value = theta
        i += 1
    return (min_value, min_index+1)


def create_vector_x_new(vector_x, vector_z, vector_jb, j0, theta, m):
    vector_x_new = np.zeros_like(vector_x)

    for i in range(m):
        index = vector_jb[i] - 1
        vector_x_new[index] = vector_x[index] - theta * vector_z[i]

    vector_x_new[j0-1] = theta

    return vector_x_new


def test1():
    A = np.array([
        [0.0, 1.0, 4.0, 1.0, 0.0, -8.0, 1.0, 5.0],
        [0.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 2.0, -1.0, 0.0, -1.0, 3.0, -1.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 0.0, 3.0, 1.0, 1.0]
    ])
    b = np.array([36.0, -11.0, 10.0, 20.0])
    c = np.array([-5.0, 2.0, 3.0, -4.0, -6.0, 0.0, 1.0, -5.0])
    x_0 = np.array([4.0, 5.0, 0.0, 6.0, 0.0, 0.0, 0.0, 5.0])
    Jb = [1, 2, 4, 8]
    correct_result = ([0.0, 9.5, 5.333, 1.5, 0.0, 0.0, 3.667, 0.0])

    # simplex(A, c, x_0, Jb)
    print('Correct answer: ', correct_result)
    return A, b, c, x_0, Jb


def test2():
    A = np.array([
        [0.0, 1.0, 1.0, 1.0, 0.0, -8.0, 1.0, 5.0],
        [0.0, -1.0, 0.0, -7.5, 0.0, 0.0, 0.0, 2.0],
        [0.0, 2.0, 1.0, 0.0, -1.0, 3.0, -1.4, 0.0],
        [1.0, 1.0, 1.0, 1.0, 0.0, 3.0, 1.0, 1.0]
    ])
    b = np.array([15.0, -45.0, 1.8, 19.0])
    c = np.array([-6.0, -9.0, -5.0, 2.0, -6.0, 0.0, 1.0, 3.0])
    x_0 = np.array([4.0, 0.0, 6.0, 6.0, 0.0, 0.0, 3.0, 0.0])
    Jb = [1, 3, 4, 7]
    correct_result = ([0.0, 0.0, 0.0, 7.055, 0.0, 1.803, 2.578, 3.958])

    # simplex(A, c, x_0, Jb)
    print('Correct answer: ', correct_result)
    return A, b, c, x_0, Jb


def test3():
    A = np.array([
        [0.0, -1.0, 1.0, -7.5, 0.0, 0.0, 0.0, 2.0],
        [0.0, 2.0, 1.0, 0.0, -1.0, 3.0, -1.5, 0.0],
        [1.0, -1.0, 1.0, -1.0, 0.0, 3.0, 1.0, 1.0]
    ])
    b = np.array([6.0, 1.5, 10.0])
    c = np.array([-6.0, -9.0, -5.0, 2.0, -6.0, 0.0, 1.0, 3.0])
    x_0 = np.array([4.0, 0.0, 6.0, 0.0, 4.5, 0.0, 0.0, 0.0])
    Jb = [1, 3, 5]
    correct_result = ([0.0, 0.75, 0.0, 2.682, 0.0, 0.0, 0.0, 13.432])

    # simplex(A, c, x_0, Jb)
    print('Correct answer: ', correct_result)
    return A, b, c, x_0, Jb


def test4():
    A = np.array([
        [2.0, -1.0, 1.0, -7.5, 0.0, 0.0, 0.0, 2.0],
        [4.0, 2.0, -1.0, 0.0, 1.0, 2.0, -1.0, -4.0],
        [1.0, -1.0, 1.0, -1.0, 0.0, 3.0, 1.0, 1.0]
    ])
    b = np.array([14.0, 14.0, 10.0])
    c = np.array([-6.0, -9.0, -5.0, 2.0, -6.0, 0.0, 1.0, 3.0])
    x_0 = np.array([4.0, 0.0, 6.0, 0.0, 4.0, 0.0, 0.0, 0.0])
    Jb = [1, 3, 5]
    correct_result = ([5.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0])

    # simplex(A, c, x_0, Jb)
    print('Correct answer: ', correct_result)
    return A, b, c, x_0, Jb


def test5():
    A = np.array([
        [-2.0, -1.0, 3.0, -7.5, 0.0, 0.0, 0.0, 2.0],
        [4.0, 2.0, -6.0, 0.0, 1.0, 5.0, -1.0, -4.0],
        [1.0, -1.0, 0.0, -1.0, 0.0, 3.0, 1.0, 1.0]
    ])
    b = np.array([-23.5, -24.0, 2.0])
    c = np.array([-6.0, 9.0, -5.0, 2.0, -6.0, 0.0, 1.0, 3.0])
    x_0 = np.array([0.0, 0.0, 0.0, 5.0, 4.0, 0.0, 0.0, 7.0])
    Jb = [4, 5, 8]
    correct_result = "Целевая функция неограничена сверху!"

    # simplex(A, c, x_0, Jb)
    print('Correct answer: ', correct_result)
    return A, b, c, x_0, Jb


def test6():
    A = np.array([
        [-2.0, -1.0, 1.0, -7.0, 0.0, 0.0, 0.0, 2.0],
        [4.0, 2.0, -1.0, 0.0, 1.0, 5.0, -1.0, -5.0],
        [1.0, 11.0, 0.0, 1.0, 0.0, 3.0, 1.0, 1.0]
    ])
    b = np.array([-2.0, 14.0, 4.0])
    c = np.array([-6.0, -9.0, 5.0, -2.0, 6.0, 0.0, -1.0, 3.0])
    x_0 = np.array([4.0, 0.0, 6.0, 0.0, 4.0, 0.0, 0.0, 0.0])
    Jb = [1, 2, 3]
    correct_result = ([0.0, 0.0, 26.0, 4.0, 36.0, 0.0, 0.0, 0.0])

    # simplex(A, c, x_0, Jb)
    print('Correct answer: ', correct_result)
    return A, b, c, x_0, Jb


def main():
    if 'test' in sys.argv:
        matrix_a, vector_b, vector_c, vector_x, vector_jb = test6()
        m = matrix_a.shape[0]
        all_j = {i for i in range(1, matrix_a.shape[1]+1)}
    else:             
        m, n = list(map(int, input('Введите количество строк \
                                   и столбцов: ').split()))
        all_j = {i for i in range(1, n+1)}

        matrix_a = input_matrix(m)
        vector_b = input_vector()
        vector_c = input_vector()
        vector_x = input_float_vector()
        vector_jb = input_vector()

    while True:
        matrix_ab = create_matrix_ab(matrix_a.copy(), vector_jb.copy())
        try:
            matrix_b = np.linalg.inv(matrix_ab.copy())
        except np.linalg.LinAlgError:
            print('Unbounded')
            exit()

        vector_cb = create_vector_cb(vector_c.copy(), vector_jb.copy())

        vector_u = vector_cb.dot(matrix_b.copy())

        vector_delta = vector_u.dot(matrix_a.copy()) - vector_c

        j0, min_delta = find_min_delta(vector_delta, all_j, vector_jb)

        if min_delta >= 0:
            return f'Bounded\n{" ".join(map(str, vector_x))}\n'

        j0 = vector_delta.argmin() + 1

        vector_z = matrix_b.dot(matrix_a[:, j0-1])

        theta, s = min_theta(
            vector_x.copy(), vector_z.copy(), vector_jb.copy())

        if theta == -1:
            return 'Unbounded'

        js = vector_jb[s-1]

        vector_x = create_vector_x_new(
            vector_x.copy(), vector_z.copy(), vector_jb.copy(),
            j0, theta, m)

        vector_jb[vector_jb.index(js)] = j0


if __name__ == '__main__':
    print(main())
