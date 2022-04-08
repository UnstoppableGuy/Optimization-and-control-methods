import math
import sys
import json
import random


def write_json(a, b, c, result='Error', path='data5.json'):
    try:
        with open(path, 'r') as rf:
            data = json.load(rf)
    except Exception:
        data = {}
        data['tests'] = []
    finally:
        data['tests'].append({
            'a': a,
            'b': b,
            'c': c,
            'result': result
        })
    with open('path', 'w') as outfile:
        json.dump(data, outfile)


def read_json():
    try:
        with open('data5.json', 'r') as rf:
            data = json.load(rf)
    except FileNotFoundError:
        raise FileNotFoundError('not sush file in directory')
    return data


def print_matrix(matrix):
    print('\n'.join([' '.join(list(map(str, map(int, i)))) for i in matrix]))


def generate_random_data_set(max_value, rows=0, colums=0):
    """Generate random dataset for tranport matrix task

    Args:
        max_value (int): sum for vectors
        rows (int, optional): count of rows in matrix. Defaults to 0.
        colums (int, optional): count of colums in matrix. Defaults to 0.
    """
    if rows < 2:
        rows = random.randint(2, 10)

    if colums < 2:
        colums = random.randint(2, 10)

    C = list([[random.randint(0, 10) for j in range(colums)]
              for i in range(rows)])

    a = [random.randint(0, int(max_value/rows)) for _ in range(rows - 1)]
    b = [random.randint(0, int(max_value/colums)) for _ in range(colums - 1)]
    last_a, last_b = max_value - sum(a), max_value - sum(b)
    a.insert(random.randint(0, len(a)), last_a)
    b.insert(random.randint(0, len(b)), last_b)
    # print(rows, colums, sep='\n')
    print(C, a, b, sep='\n')
    result, path = matrix_transport_task(rows, colums, C, a, b)
    if type(path) == str:
        write_json(a=a, b=b, c=C, result=result, path=path)
    # write_json(a=a, b=b, c=C, result=result)
    return result


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
    """_summary_

    Args:
        m (int): count of  rows
        n (int): count of colums
        matrix_c (lists in list): traffic matrix
        vector_a (list): supply vector
        vector_b (list): demand vector

    Returns:
        list: matrix X values
    """
    s = 0
    t = m + n + 1
    sum_a = sum(vector_a)
    sum_b = sum(vector_b)
    if sum_a != sum_b:
        print('sums not equal')
    edges = []
    [[edges.append([i + 1, m + j + 1, math.inf, matrix_c[i][j]])
      for j in range(n)] for i in range(m)]
    [edges.append([0, i + 1, vector_a[i], 0]) for i in range(m)]
    [edges.append([m + i + 1, n + m + 1, vector_b[i], 0]) for i in range(n)]
    route = min_cost_flow(n + m + 2, edges, min(sum_a, sum_b), s, t, n, m)
    # print(('\n'.join(str(x) for x in edges)
    #        ).replace('[', '').replace(']', ''))

    # return '\n'.join([' '.join(list(map(str, map(int, i)))) for i in route])
    count = 0
    for i in range(m):
        for j in range(n):
            if route[i][j] > 0:
                count += 1

    if count == m+n-1:
        print('Count: {}\n'.format(count))
    else:
        print('Count:{}\n Sum:{}\n'.format(count, m+n-1))
        return route, 'error.json'
        # raise Exception('Count and sum not equal')
    return route


if __name__ == '__main__':
    if 'test' in sys.argv:
        print(test1(), end='\n\n')
        print(test2(), end='\n\n')
        print(test3(), end='\n\n')
        print(test4(), end='\n\n')
        print(test5(), end='\n\n')
    elif 'json' in sys.argv:
        data = read_json()
        for x in data['tests']:
            a = x['a']
            b = x['b']
            c = x['c']
            res = matrix_transport_task(len(a), len(b), c, a, b)
            from numpy import array_equal
            if not array_equal(x['result'], res):
                print('RESULTT NOT EQUAL')
    elif 'random' in sys.argv:
        try:
            max_value, row, col = map(int, input().split(' '))
        except Exception:
            max_value = 500
            row = 5
            col = 5
        finally:
            print_matrix(generate_random_data_set(max_value, row, col))
        # generate_random_data_set(500, 5, 6)
    else:
        m, n = map(int, input().split(' '))
        matrix_c = input_matrix(m)
        vector_a = input_vector()
        vector_b = input_vector()
