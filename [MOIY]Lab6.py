import numpy as np
import math

A = np.matrix([[1, 2, 1, 0, 0, 4, -1, -3],
               [1, 3, 0, 1, 0, -1, -1, 2],
               [1, 4, 0, 0, 1, 2, -2, 0]])
D = np.matrix([[6, 11, -1, 0, 6, -7, -3, -2],
               [11, 41, -1, 0, 7, -24, 0, -3],
               [-1, -1, 1, 0, -3, -4, 2, -1],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [6, 7, -3, 0, 11, 6, -7, 1],
               [-7, -24, -4, 0, 6, 42, -7, 10],
               [-3, 0, 2, 0, -7, -7, 5, -1],
               [-2, -3, -1, 0, 1, 10, -1, 3]])
b = np.matrix([[4, 5, 6]])
c = np.matrix([[-10], [-31], [7], [0], [-21], [-16], [11], [-7]])
J_on = [2, 3, 4]
J_ = [2, 3, 4]
x = np.matrix([[0], [0], [6], [4], [5], [0], [0], [0]])



def end_method_for_quadratic_task(A, D, b, c, optimal_indexes, right_optimal_indexes, optimal_plan):
    iteration = 0
    while True:
        print('\n------- Итерация № {} -------'.format(iteration))

        a_b_matrix = A[:, optimal_indexes]
        inverse_a_b_matrix = np.linalg.inv(a_b_matrix)

        print('\nБазисная матрица А:\n {}\n'.format(a_b_matrix))
        print('Обратная бащисная матрица А:\n {}\n'.format(inverse_a_b_matrix))

        vector_c_x = c + D * optimal_plan
        print("Вектор с: \n{}".format(vector_c_x))

        c_b = np.matrix([[vector_c_x.item(i)] for i in optimal_indexes])
        print("\nБазисный вектор с: \n{}".format(c_b))

        u = -c_b.transpose() * inverse_a_b_matrix
        print("\nВектор потенциалов u: {}".format(u))

        delta = u * A + vector_c_x.transpose()
        print('\nВектор оценок: {}'.format(delta))

        negative_elements_indexes = get_negative_elements_in_list_indexes(delta.tolist()[0], optimal_indexes)
        print("\nИндексы негативных элементов: {}".format(negative_elements_indexes))

        if len(negative_elements_indexes) == 0:
            break

        j0 = negative_elements_indexes[0]
        print('j0: {}\n'.format(j0))

        delta_j0 = delta.item(j0)
        l = calculate_l(A, right_optimal_indexes, j0, delta_j0, D)
        print("Вектор l: {}".format(l))

        min_theta_index, min_theta = calculate_thetas(l, D, optimal_plan, j0, delta_j0, right_optimal_indexes)
        print("\nj*: {}\n".format(min_theta_index))

        optimal_plan = optimal_plan + min_theta * l.transpose()
        print("\nТекущий оптимальный план:\n {}".format(optimal_plan))

        update_indexes(optimal_indexes, right_optimal_indexes, j0, min_theta_index, inverse_a_b_matrix, A)

        print("\nОпора ограничений: {}".format(optimal_indexes))
        print("\nПравильные оптимальные индексы: {}".format(right_optimal_indexes))

        iteration += 1
    return optimal_plan, optimal_indexes, right_optimal_indexes


def get_negative_elements_in_list_indexes(elements_list, basis_indexes):
    not_basis_indexes = [index for index in range(len(elements_list)) if index not in basis_indexes]
    indexes = [not_basis_indexes[index] for index in range(len(not_basis_indexes))
               if elements_list[not_basis_indexes[index]] < -0.33]
    return indexes


def calculate_l(a_matrix, right_optimal_indexes, j0, delta_j0, d_matrix):
    sign = lambda x: (1, -1)[x < 0]

    elements_count = len(a_matrix.tolist()[0])
    l = [0] * elements_count
    l[j0] = -sign(delta_j0)

    a_b = a_matrix[:, right_optimal_indexes].tolist()

    indexes_count = len(right_optimal_indexes)

    d_b = [[0] * indexes_count for _ in range(indexes_count)]
    d_matrix = d_matrix.tolist()
    i = 0
    j = 0
    for first_index in right_optimal_indexes:
        for second_index in right_optimal_indexes:
            d_b[i][j] = d_matrix[first_index][second_index]
            j += 1

        i += 1
        j = 0

    print("d*: {}".format(d_b))

    h = [[0] * (len(a_b) + indexes_count) for _ in range(len(a_b) + indexes_count)]

    for i in range(len(h)):
        for j in range(len(h[i])):
            if i < indexes_count and j < indexes_count:
                h[i][j] = d_b[i][j]

            if i >= indexes_count > j:
                h[i][j] = a_b[i - indexes_count][j]

            if i < indexes_count <= j:
                h[i][j] = a_b[j - indexes_count][i]

    h = np.matrix(h)
    print("Матрица h:\n {}".format(h))

    b = [0] * indexes_count
    for index in range(indexes_count):
        b[index] = d_matrix[right_optimal_indexes[index]][j0]

    for element in a_matrix[:, j0].tolist():
        b.append(element[0])

    b = np.matrix(b)

    print("\nВектор b: {}".format(b))

    x = -np.linalg.inv(h) * b.transpose()
    print("Вектор x:\n {}".format(x))

    i = 0
    for optimal_index in right_optimal_indexes:
        l[optimal_index] = x.item(i)
        i += 1

    return np.matrix(l)


def calculate_thetas(l, d_matrix, optimal_plan, j0, delta_j0, right_optimal_indexes):
    delta = (l * d_matrix * l.transpose()).item(0)
    print("\ndelta: {}".format(delta))

    thetas = {j0: math.inf if delta == 0 else math.fabs(delta_j0) / delta}

    for optimal_index in right_optimal_indexes:
        thetas[optimal_index] = (-optimal_plan.item(optimal_index) / l.item(optimal_index)
                                 if l.item(optimal_index) < 0 else math.inf)

    min_theta_index = min(thetas, key=thetas.get)
    min_theta = thetas[min_theta_index]

    if min_theta == math.inf:
        raise Exception

    return min_theta_index, min_theta


def update_indexes(optimal_indexes, right_optimal_indexes, j0, min_theta_index, inverse_a_b, a_matrix):
    if j0 == min_theta_index:
        right_optimal_indexes.append(min_theta_index)
        return

    right_optimal_indexes_not_in_optimal_indexes = [index for index in right_optimal_indexes
                                                    if index not in optimal_indexes]

    if min_theta_index in right_optimal_indexes_not_in_optimal_indexes:
        right_optimal_indexes.remove(min_theta_index)

    if min_theta_index in optimal_indexes:
        s = optimal_indexes.index(min_theta_index)

        exists = False
        j_plus = -1
        for index_not_in_optimal_indexes in right_optimal_indexes_not_in_optimal_indexes:
            value = inverse_a_b * a_matrix[:, index_not_in_optimal_indexes]

            if value.item(s) != 0:
                exists = True
                j_plus = index_not_in_optimal_indexes
                break

        if exists:
            right_optimal_indexes.remove(min_theta_index)
            optimal_indexes[s] = j_plus

            return

        optimal_indexes[s] = j0
        min_theta_index_in_right_optimal_indexes = right_optimal_indexes.index(min_theta_index)
        right_optimal_indexes[min_theta_index_in_right_optimal_indexes] = j0


x, J_on, J_ = end_method_for_quadratic_task(A, D,
                                            b, c, J_on,
                                            J_,
                                            x)

print("\nКонечный метод:")
print("Оптимальный план:\n {}\nОпорные индексы {}\nПравильные оптимальные индексы: {}".format(x.transpose(),
                                                                                  J_on,
                                                                                  J_))

