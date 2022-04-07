import math
import sys


def test1():
    """test 1

    Returns:
        str: transportation plan
    """
    a = [100, 300, 300]
    b = [300, 200, 200]
    c = [[8, 4, 1],
         [8, 4, 3],
         [9, 7, 5]]
    # x, J = solve_mtp(a, b, c)
    return matrix_transport_task(3, 3, c, a, b)


def test2():
    """test 2

    Returns:
        str: transportation plan
    """
    a = [20, 30, 25]
    b = [10, 10, 10, 10, 10]
    c = [[2, 8, -5, 7, 10],
         [11, 5, 8, -8, -4],
         [1, 3, 7, 4, 2]]

    # x, J = solve_mtp(a, b, c)
    return matrix_transport_task(3, 5, c, a, b)


def test3():
    """test 3

    Returns:
        str: transportation plan
    """
    a = [20, 11, 18, 27]
    b = [11, 4, 10, 12, 8, 9, 10, 4]
    c = [[-3, 6, 7, 12, 6, -3, 2, 16],
         [4, 3, 7, 10, 0, 1, -3, 7],
         [19, 3, 2, 7, 3, 7, 8, 15],
         [1, 4, -7, -3, 9, 13, 17, 22]]

    # x, J = solve_mtp(a, b, c)
    return matrix_transport_task(4, 8, c, a, b)


def test4():
    """test 4

    Returns:
        str: transportation plan
    """
    a = [15, 12, 18, 20]
    b = [5, 5, 10, 4, 6, 20, 10, 5]
    c = [[-3, 10, 70, -3, 7, 4, 2, -20],
         [3, 5, 8, 8, 0, 1, 7, -10],
         [-15, 1, 0, 0, 13, 5, 4, 5],
         [1, -5, 9, -3, -4, 7, 16, 25]]

    # x, J = solve_mtp(a, b, c)
    return matrix_transport_task(4, 8, c, a, b)


def test5():
    """test 5

    Returns:
        str: transportation plan
    """
    a = [53, 20, 45, 38]
    b = [15, 31, 10, 3, 18]
    c = [[3, 0, 3, 1, 6],
         [2, 4, 10, 5, 7],
         [-2, 5, 3, 2, 9],
         [1, 3, 5, 1, 9]]

    # x, J = solve_mtp(a, b, c)
    return matrix_transport_task(4, 5, c, a, b)


def input_matrix(size):
    """enter matrix

    Args:
        size (int): size for matrix

    Returns:
        list: matrix with args
    """
    matrix = list([[int(j) for j in input("Строка матрицы: ").split()]
                   for i in range(size)])
    return matrix


def input_vector():
    """enter vector

    Returns:
        list: vector with args
    """
    return list(map(int, input('Элементы вектора: ').split()))


def shortest_paths(n, v0, adj, capacity, cost):
    vector_d = [math.inf for _ in range(n)]
    vector_d[v0] = 0
    inq = [False for _ in range(n)]
    q = [v0]
    p = [-1 for _ in range(n)]

    while len(q):
        u = q[0]
        del q[0]
        inq[u] = False
        for v in adj[u]:
            if capacity[u][v] > 0 and vector_d[v] > vector_d[u] + cost[u][v]:
                vector_d[v] = vector_d[u] + cost[u][v]
                p[v] = u
                if not inq[v]:
                    inq[v] = True
                    q.append(v)

    return vector_d, p


def min_cost_flow(N, edges, K, s, t, n, m):
    adj = [[] for _ in range(N)]
    cost = [[0 for _ in range(N)] for _ in range(N)]
    capacity = [[0 for _ in range(N)] for _ in range(N)]

    for e in edges:
        adj[e[0]].append(e[1])
        adj[e[1]].append(e[0])
        cost[e[0]][e[1]] = e[3]
        cost[e[1]][e[0]] = -e[3]
        capacity[e[0]][e[1]] = e[2]

    flow = 0
    cost_s = 0

    while flow < K:
        vector_d, p = shortest_paths(N, s, adj, capacity, cost)
        if vector_d[t] == math.inf:
            break

        f = K - flow
        cur = t
        while cur != s:
            f = min(f, capacity[p[cur]][cur])
            cur = p[cur]

        flow += f
        cost_s += f * vector_d[t]
        cur = t

        while cur != s:
            capacity[p[cur]][cur] -= f
            capacity[cur][p[cur]] += f
            cur = p[cur]

    ans = [[0 for _ in range(n)] for _ in range(m)]
    for i in range(m):
        for j in range(n):
            ans[i][j] = capacity[j + m + 1][i + 1]

    if flow < K:
        return None
    else:
        return ans


def matrix_transport_task(m, n, matrix_c, vector_a, vector_b):
    s = 0
    t = m + n + 1
    sum_a = sum(vector_a)
    sum_b = sum(vector_b)
    edges = []

    for i in range(m):
        for j in range(n):
            edges.append([i + 1, m + j + 1, math.inf, matrix_c[i][j]])

    for i in range(m):
        edges.append([0, i + 1, vector_a[i], 0])

    for i in range(n):
        edges.append([m + i + 1, n + m + 1, vector_b[i], 0])

    route = min_cost_flow(n + m + 2, edges, min(sum_a, sum_b), s, t, n, m)
    print(('\n'.join(str(x) for x in edges)
           ).replace('[', '').replace(']', ''))
    print(route)
    return '\n'.join([' '.join(list(map(str, map(int, i)))) for i in route])


if __name__ == '__main__':
    if 'test' in sys.argv:
        print(test1())
        # print(test2())
        # print(test3())
        # print(test4())
        # print(test5())
    else:
        print(matrix_transport_task())
