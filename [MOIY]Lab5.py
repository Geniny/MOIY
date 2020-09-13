import numpy.linalg


a = [500, 300, 100]
b = [150, 350, 200, 100, 100]
C = [[3, 3, 5, 3, 1],
     [4, 3, 2, 4, 5],
     [3, 7, 5, 4, 2]]

def nord_west_method(A, b):
    n = len(b)
    m = len(A)
    X = [[0] * n for _ in range(m)]
    i, j = 0, 0
    I_b = []

    while True:
        I_b.append((i, j))
        max_supply = min(A[i], b[j])
        A[i] -= max_supply
        b[j] -= max_supply
        X[i][j] = max_supply
        if i == m - 1 and j == n - 1:
            break
        if A[i] == 0 and i != len(A) - 1:
            i += 1
        elif b[j] == 0 and j != len(b) - 1:
            j += 1

    return X, I_b


def get_potentials(C, I_b):
    m = len(C)
    n = len(C[0])
    A = [[0] * (n + m) for _ in range(n + m)]
    b = [0] * (n + m)

    A[-1][0] = 1
    b[-1] = 0

    for it_num, (i, j) in enumerate(I_b):
        A[it_num][i] = A[it_num][m + j] = 1
        b[it_num] = C[i][j]

    x = numpy.linalg.solve(A, b)
    return x[:m], x[m:]


def rebuild_I_b(m, n, I_b):
    I_bh, I_bv = [[] for _ in range(m)], [[] for _ in range(n)]

    for i, j in I_b:
        I_bh[i].append(j)
        I_bv[j].append(i)
    return I_bh, I_bv


def get_I_b_matrix(m, n, I_b):
    I_bm = [[0] * n for _ in range(m)]

    for i, j in I_b:
        I_bm[i][j] = 1

    return I_bm


def i_j_generator(n, m):
    for i in range(n):
        for j in range(m):
            yield i, j


def transport_task_solver(a, b, C):
    m = len(a)
    n = len(b)

    print("\nПроверка на замкнутость: ")
    difference = sum(a) - sum(b)
    if difference != 0:
        print("- Система не замкнута")
        if difference > 0:
            b.append(difference)
            n += 1
            for row in C:
                row.append(0)
        elif difference < 0:
            a.append(-difference)
            m += 1
            C.append([0] * n)
    else:
        print("- Система замкнута")

    print("\nВектор a: ", a)
    print("Вектор b:", b)
    print("Матрица C: ", C)

    print("\nНачальный опорный план: ")
    X, I_b = nord_west_method(a, b)
    print("Базисный план: ", X)
    print("Множество клеток: ", I_b)

    print("\nМетод потенциалов: ")
    iteration = 0
    while True:
        print("\n------- Итерация № {} -------".format(iteration))
        u, v = get_potentials(C, I_b)
        I_bm = get_I_b_matrix(m, n, I_b)

        for i, j in i_j_generator(m, n):
            if I_bm[i][j] == 0 and u[i] + v[j] > C[i][j]:
                I_bm[i][j] = 1
                I_b.append((i, j))
                break
        else:
            print("\nИтоговый оптимальный план: ")
            for i0 in range(m):
                for j0 in range(n):
                    print(X[i0][j0], end='\t')
                print()
            return
        print("\nРасширенное множество клеток: ", I_b)

        I_bh, I_bv = rebuild_I_b(m, n, I_b)
        print("Клетки на концах горизонтальных звеньев: ", I_bh)
        loop = []
        loop.append((i, j))
        deleted = True

        while deleted:
            deleted = False
            for i, row in enumerate(I_bh):
                if len(row) < 2:
                    for j in row:
                        I_bv[j].remove(i)
                        deleted = True
                    row.clear()
            for j, column in enumerate(I_bv):
                if len(column) < 2:
                    for i in column:
                        I_bh[i].remove(j)
                        deleted = True
                    column.clear()
        up = True
        i, j = loop[0]
        while True:
            if up:
                up = False
                i = I_bv[j][1] if I_bv[j][0] == i else I_bv[j][0]
            else:
                up = True
                j = I_bh[i][1] if I_bh[i][0] == j else I_bh[i][0]
            if i == loop[0][0] and j == loop[0][1]:
                break
            else:
                loop.append((i, j))

        theta = min(X[loop[i][0]][loop[i][1]] for i in range(1, len(loop), 2))
        factor = 1
        for i, j in loop:
            X[i][j] += factor * theta
            if X[i][j] == 0 and len(I_b) > n + m - 1 and factor == -1:
                I_b.remove((i, j))
            factor = -1 if factor == 1 else 1
        print("Текущий план X: ", X)
        iteration += 1

transport_task_solver(a, b, C)
