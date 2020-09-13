import numpy as np

A = np.array([[1., 1., 1.,],
              [2., 2., 2.,]])

b = np.array([0.,0.])

def basis_matrix(a, jb):
    Ab = [[element for j, element in enumerate(row) if j + 1 in jb] for row in list(a)]
    return Ab

def simplex_main_phase(A, x, all_c, Jb, Jn, B, iteration = 0):
    print("\n------- Итерация №", iteration, ' -------')

    c = np.array([all_c[j - 1] for j in Jb])
    print("\nВектор базисных коэффициентов с:", c)

    u = np.dot(c.transpose(), B)
    print("\nВектор потециалов u:", u)

    delta = np.matmul(u, A) - all_c
    print("\nВектор оценок delta: ", delta)

    if (all([i >= 0 for i in delta])):
        print("Решение найдено: ", x)
        return x, Jb

    min_component_array = [(delta[j - 1], j) for j in Jn]
    min_component, j0 = min(min_component_array, key=lambda x: x[0])
    print("\nМинимальная оценка из небазисных индексов: ", min_component)
    print("Индекс: ", j0)

    print("\nВектор z: ")
    z = np.matmul(B, A[:, j0 - 1].reshape(B.shape[1], 1))
    print(z)

    if all(i <= 0 for i in z):
        print("Все элементы вектора z не положительны")
        return np.inf

    Theta = [(float(x[Jb[i] - 1] / z[i]), i + 1) for i, z_element in enumerate(z) if z_element > 0]
    print("\nВектор θ: ", Theta)
    Theta_min, index = min(Theta, key=lambda x: x[0])
    print("θ{} = {}".format(index, float(Theta_min)))

    new_x = []
    new_Jb = list(Jb)
    new_Jb[index - 1] = j0
    J_tmp = list(Jn)
    J_tmp.extend(Jb)
    new_Jn = np.array([elem for elem in J_tmp if elem not in new_Jb])

    for i, old_x in enumerate(x):
        if i + 1 in list(filter(lambda x: x != j0, Jn)):
            new_x.append(0)
        elif i + 1 == j0:
            new_x.append(float(Theta_min))
        else:
            j = list(Jb).index(i + 1)
            new_x.append(float(x[Jb[j] - 1] - Theta_min * z[j]))

    print("\nНовый базисный план:", new_x)
    print("Новыe базисные индексы:", new_Jb)
    print("Новые не базисные индексы:", new_Jn)

    ds = []
    for i, z_i in enumerate(z):
        if i + 1 == index:
            ds.append(float(1 / z[index - 1]))
        else:
            ds.append(float(-z_i / z[index - 1]))

    print("ds =",ds)
    M = np.eye(len(B))
    for i in range(len(B)):
        M[i][index - 1] = ds[i]

    new_B = np.dot(M, B)
    print("\nНовая обратная матрица:")
    print(new_B)
    return simplex_main_phase(A, new_x, all_c, new_Jb, new_Jn, new_B, iteration + 1)

def simplex_first_phase(A, b):
    m, n = A.shape
    for i, bi in enumerate(b):
        if bi < 0:
            A[i] = -A[i]
            b[i] = -bi

    E = np.identity(m)
    A_extended = np.concatenate((A, E), axis=1)

    Jn = np.array(range(1, n + 1))
    start_Jb = np.array(range(n + 1, n + m + 1))
    x = np.concatenate((np.zeros(n), b), axis = 0)
    c = np.array([0] * n + [-1] * m)

    B = np.linalg.inv(basis_matrix(A_extended, start_Jb))
    optimal_plan, Jb = simplex_main_phase(A_extended, x, c, start_Jb, Jn, B)

    if not all(xu == 0 for xu in optimal_plan[n:]):
        print("\nИскуственные переменные не равны нулю")
        return
    else:
        print("\nИскуственные переменные равны нулю")

    phantom_indexes = set(Jb).intersection(start_Jb)
    if phantom_indexes == set():
        return optimal_plan[:n], Jb

    while len(phantom_indexes) != 0:
        j_k = next(iter(phantom_indexes))
        k = Jb.index(j_k)
        relative_index = j_k - n

        found = False
        Jn = set(Jn).difference(Jb)
        for jni in Jn:
            B = np.linalg.inv(basis_matrix(A_extended, Jb))
            alpha = B.dot(A[:, jni - 1])
            if alpha[k] != 0:
                Jb[k] = jni
                found = True

        if not found:
            A = np.delete(A, relative_index)
            A_extended = np.delete(A_extended, relative_index)
            Jb.remove(Jb[k])
            phantom_indexes.remove(j_k)

simplex_first_phase(A, b)