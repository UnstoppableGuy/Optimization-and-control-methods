import numpy as np
import random


def generation_random_data():
    size = random.randint(2, 5)
    index = random.randint(2, size)
    A = np.array([[random.randint(0, 100) for j in range(size)]
                 for i in range(size)])
    vector = np.array([random.randint(0, 100) for j in range(size)])
    return size, index, A, vector


def input_matrix(size):
    matrix = np.array([[float(j) for j in input("Строка матрицы: ").split()]
                       for i in range(size)])
    return matrix


def input_vector(size):
    return np.array([float(j) for j in input('Элементы вектора: ').split()])


def mofidy_output_matrix(matrix):
    answer = []
    for el in matrix:
        answer.append(' '.join(map(str, el)))
    return '\n'.join(answer)


def main():
    # n = int(input("Размер матрицы: "))
    # i = int(input("Заменяемый столбец: "))
    # # matrix_a = input_matrix(n)
    # matrix_b = input_matrix(n)
    # vector_x = input_vector(n)
    # vector_z = matrix_b.dot(vector_x)

    # test config
    # n, i, matrix_b, vector_x = generation_random_data()
    # vector_z = matrix_b.dot(vector_x)

    matrix_b = np.array([[-24., 20., -5.],
                         [18., -15., 4.],
                         [5., - 4., 1.]], dtype=float)

    vector_x = np.array([2., 2., 2.])

    vector_z = matrix_b.dot(vector_x)
    i = 2
    n = 3

    if vector_z[i - 1] == 0:
        return 'Необратима'
    vector_l = vector_z.copy()
    vector_l[i - 1] = -1

    vector_l_cover = -1 / vector_z[i - 1] * vector_l

    matrix_m = np.eye(n)
    matrix_m[:, i - 1] = vector_l_cover

    return f'Обратима\n{mofidy_output_matrix(matrix_m.dot(matrix_b))}'


if __name__ == '__main__':
    # while True:
    print(main())
