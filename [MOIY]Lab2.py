import numpy as np

A = np.array([[1., 2., 1., 0., 0., 0.],
              [2., 1., 0., 1., 0., 0.],
              [1., 0., 0., 0., 1., 0.],
              [0., 1., 0., 0., 0., 1.]])

b = np.array([10., 11., 5., 4.])

c = np.array([20., 26., 0., 0., 0., 1.])

x = np.array([2., 4., 0., 3., 3., 0.])
Jb = np.array([5,2,1,4])


def basis_matrix(A, Jb):
    Jb_count = len(Jb)
    Ab = np.arange(Jb_count ** 2, dtype=float).reshape((Jb_count, Jb_count))
    for i in range(Jb_count):
        for j in range(Jb_count):
            Ab[i][j] = A[j][Jb[i] - 1]
    return Ab


def simplex_main_phase(c, A, B, x, Jb, iteration = 0):
    print("\n------- Итерация №",iteration,' -------')
    Ab = basis_matrix(A, Jb)
    Jn = np.setdiff1d(np.arange(1, A.shape[1] + 1, 1), Jb)

    print("\nБазисный план Х: ")
    print(x)

    print("\nБазисная матрица Ab: ")
    print(Ab)

    if (B is None):
        B = np.linalg.inv(Ab)
    print("\nОбратная базисная матрица Ab: ")
    print(B)

    cb = np.array([c[j-1] for j in Jb])
    print("\nБазисные вектор сb:")
    print(cb)

    u = np.matmul(cb, Ab)
    print("\nВектор потенциалов u:")
    print(u)

    delta = np.matmul(u,A) - c
    print("\nВектор оценок delta:")
    print(delta)

    if(all([i >= 0 for i in delta])):
        print("Решение найдено: ")
        print(x)
        return x, Jb

    print("\nМинимальная оценка из небазисных индексов: ", end = '')
    min_component_array = [(delta[j-1], j)  for j in Jn]
    min_component = min(min_component_array, key=lambda x: x[0])
    print(min_component[0])
    print("Индекс: ", end = '')
    print(min_component[1])

    print("\nВектор z: ")
    z = np.matmul(B, A[:,min_component[1]-1].reshape(B.shape[1], 1))
    print(z)

    print("\nВектор θ: ")
    Theta =  [(x[Jb[i] - 1] / z_element, i + 1) for i, z_element in enumerate(z) if z_element > 0]
    Theta_min, index = min(Theta, key=lambda x: x[0])
    print("θ{} = {}".format(index, float(Theta_min)))

    new_x = []
    new_Jb = list(Jb)
    new_Jb[index - 1] = min_component[1]
    J_tmp = list(Jn)
    J_tmp.extend(Jb)
    new_Jn = np.array([elem for elem in J_tmp if elem not in new_Jb])

    for i, old_x in enumerate(x):
        if i + 1 in list(filter(lambda x: x != min_component[1], Jn)):
            new_x.append(0)
        elif i + 1 == min_component[1]:
            new_x.append(float(Theta_min))
        else:
            j = list(Jb).index(i + 1)
            new_x.append(float(x[Jb[j] - 1] - Theta_min * z[j]) )

    print("\nНовый базисный план:", new_x)
    print("Новыe базисные индексы:", new_Jb)
    print("Новые не базисные индексы:", new_Jn,'\n')

    ds = []
    for i, z_i in enumerate(z):
        if i + 1 == index:
            ds.append(float(1 / z[index - 1]))
        else:
            ds.append(float(-z_i / z[index - 1]))

    print("ds =", ds)
    M = np.eye(len(B))
    for i in range(len(B)):
        M[i][index - 1] = ds[i]

    new_B = np.dot(M, B)
    print("Новая обратная матрица:")
    print(new_B)

    simplex_main_phase(c, A, new_B, new_x,  new_Jb, iteration + 1)

simplex_main_phase(c, A, None, x, Jb)

