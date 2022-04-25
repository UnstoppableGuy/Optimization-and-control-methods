import sys
import itertools
import numpy as np
from lab5 import input_matrix, input_vector


def print_matrix(matrix):
    return(f'\n'.join([' '.join(list(map(str, map(int, i)))) for i in matrix]))


def square_task(c, D, A, x, J, J_adv):
    m = len(A)
    n = len(A[0])
    itr = 1
    while True:
        print(f'\nIteration {itr}:\n')

        # STEP 1
        print('X:\t' + np.array2string(x))
        cx = c + x @ D
        print(f'Cx:\t{np.array2string(cx)}')
        cbx = np.array([cx[J[i] - 1] for i in range(len(J))])
        A_b = np.array([(A[:, J[i] - 1]) for i in range(len(J))])
        A_b_inv = np.linalg.inv(A_b)
        print(f'A_b_inv:\n{print_matrix(A_b_inv)}')
        ux = -cbx @ A_b_inv
        print(f'ux:\t{ux}')
        deltax = ux @ A + cx
        print(f'deltax:\t{deltax}')

        # STEP 2
        if min(deltax) >= 0:
            return f'ANSWER:\t{x}'  # return

        # STEP 3
        j0 = list(deltax).index(min(deltax))  # python index
        print(f'j0:\t{j0}')
        # STEP 4

        vector_l = np.zeros(n)
        vector_l[j0] = 1
        print(f'J_adv:\t{J_adv}')
        l_adv = np.delete(vector_l, J_adv - 1, axis=0)
        print(f'l_adv:\t{l_adv}')

        # STEP 4.a
        # all linear combinations of J(coordinates of Ds elements)
        D_adv_indx = list(
            itertools.combinations_with_replacement(J_adv - 1, 2))
        lJ = len(J_adv)
        D_adv = np.zeros([lJ, lJ])
        k = 0
        for i in range(lJ):
            for j in range(i, lJ):
                if D_adv_indx[k][0] != D_adv_indx[k][1]:
                    D_adv[j][i] = D[D_adv_indx[k][1]][D_adv_indx[k][0]]
                D_adv[i][j] = D[D_adv_indx[k][0]][D_adv_indx[k][1]]
                k += 1

        A_adv_b = np.array([(A[:, J_adv[i] - 1]) for i in range(len(J_adv))])
        At_adv_b = np.transpose(A_adv_b)
        print(f'D_adv:\n{print_matrix(D_adv)}')
        print(f'A_adv_b:\n{print_matrix(A_adv_b)}')

        Matrix1 = np.row_stack((D_adv, At_adv_b))
        Matrix2 = np.row_stack(
            (A_adv_b, np.zeros([len(A_adv_b[0]), len(At_adv_b)])))
        H = np.column_stack((Matrix1, Matrix2))

        # STEP 4.b
        b_up = np.array([D[:, j0][J_adv[i] - 1] for i in range(len(J_adv))])
        print(f'b_up:\t{b_up}')
        b_down = A[:, j0]
        print(f'b_down:\t{b_down}')
        b = np.concatenate((b_up, b_down))

        # STEP 4.c
        print(f'H:\n{print_matrix(H)}')
        print(f'b:\t{b}')
        x_hb = -np.linalg.inv(H) @ b
        print(f'x_hb:\t{x_hb}')
        ladv = np.array([x_hb[i] for i in range(len(J_adv))])
        print(f'ladv:\t{ladv}')
        vector_l = np.concatenate((ladv, l_adv))

        # STEP 5
        print(f'vector_l:\t{vector_l}')
        print(f'D:\n{print_matrix(D)}')

        delta = vector_l @ D @ vector_l[:, np.newaxis]
        print(f'delta:\t{delta}')

        teta = np.full(len(J_adv), np.inf)
        teta_j0 = np.inf
        if delta > 0:
            teta_j0 = abs(deltax[j0]) / delta
        print(f'teta_j0:\t{teta_j0}')

        for i in range(len(teta)):
            if vector_l[i] < 0:
                teta[i] = -x[i] / vector_l[i]
        teta = np.append(teta, teta_j0)
        print(f'teta:\t{teta}')

        teta0 = min(teta)
        print(f'teta0:\t{teta0}')

        if teta0 == np.inf or teta0 > 1e+16:
            return ('TARGET FUNCTION IS UNBOUNDED')  # return

        js = j0  # j*
        if teta0 != teta_j0:
            js = J_adv[list(teta).index(teta0)] - 1  # python index
        print(f'js:\t{js}')

        # STEP 6(UPDATE)
        x = x + teta0 * vector_l
        print(f'teta0 * vector_l:\t{teta0 * vector_l}')
        print(f'J:\t{J}\nJ_adv:\t{J_adv}')

        last_condition = True
        if js == j0:
            J_adv = np.append(J_adv, js + 1)
            last_condition = False
        elif js + 1 in J_adv and js+1 not in J:
            J_adv = np.delete(J_adv, np.where(J_adv == js + 1))
            last_condition = False
        elif js + 1 in J:
            s = list(J).index(js + 1)
            J_adv_tmp = set(J_adv) - set(J)
            J_adv_tmp = list(J_adv_tmp)
            print(f'J_adv_tmp:\t{J_adv_tmp}')
            for i in range(len(J_adv_tmp)):
                j_plus = J_adv_tmp[i]  # not python index
                vector_tmp = A_b_inv @ A[:, j_plus - 1]
                print(vector_tmp)
                if vector_tmp[s] != 0:
                    J = np.where(J == js + 1, j_plus, J)
                    J_adv = np.delete(J_adv, np.where(J_adv == j_plus))
                    last_condition = False
                    break

        if last_condition:
            J = np.where(J == js + 1, j0 + 1, J)
            J_adv = np.where(J_adv == js + 1, j0 + 1, J_adv)
        itr += 1


def test1():
    c = np.array([0, -1, 0])
    D = np.array([[2, -1, 0],
                  [-1, 2, -1],
                  [0, -1, 2]])
    A = np.array([[2, 1, 0],
                  [0, 1, 2]])
    x = np.array([0, 2, 1])
    J = np.array([2, 3])
    J_adv = np.array([2, 3])
    print(square_task(c, D, A, x, J, J_adv))


def test2():

    c = np.array([0, 0, -2])
    D = np.array([[1, 0, 0],
                  [0, 1, -1],
                  [0, -1, 2]])
    A = np.array([[0, 1, 1],
                  [1, 0, 1]])
    x = np.array([2, 4, 0])
    J = np.array([1, 2])
    J_adv = np.array([1, 2])
    print(square_task(c, D, A, x, J, J_adv))


def test0():
    c = np.array([-8, -6, -4, -6])
    D = np.array([[2, 1, 1, 0],
                  [1, 1, 0, 0],
                  [1, 0, 1, 0],
                  [0, 0, 0, 0]])
    A = np.array([[1, 0, 2, 1],
                  [0, 1, -1, 2]])
    x = np.array([2, 3, 0, 0])
    J = np.array([1, 2])
    J_adv = np.array([1, 2])
    print(square_task(c, D, A, x, J, J_adv))


def test4():
    A = np.array([[11, 0, 0, 1, 0, -4, -1, 1], [1, 1, 0, 0,
                 1, -1, -1, 1], [1, 1, 1, 0, 1, 2, -2, 1]])
    B = np.array([[1, -1, 0, 3, -1, 5, -2, 1], [2, 5, 0, 0, -
                 1, 4, 0, 0], [-1, 3, 0, 5, 4, -1, -2, 1]])
    b = np.array([8, 2, 5])
    j = np.array([1, 2, 3])
    je = np.array([1, 2, 3])
    x = np.array([])


if __name__ == '__main__':
    if 'test' in sys.argv:
        test0()
        # test1()
        # test2()
    else:
        m, n = map(int, input().split())
        print('matrix a:')
        matrix_a = np.array(input_matrix(m))
        print('vector b:')
        vector_b = np.array(input_vector())
        print('vector c:')
        vector_c = np.array(input_vector())
        print('matrix D:')
        matrix_d = np.array(input_matrix(n))
        print('vector X:')
        vector_x = np.array(input_vector())
        print('vector j')
        vector_j = np.array(input_vector())
        print('vector je')
        vector_je = np.arange(input_vector())
        square_task(c=vector_c, A=matrix_a, J=vector_j,
                    J_adv=vector_je, D=matrix_d, x=vector_x)
