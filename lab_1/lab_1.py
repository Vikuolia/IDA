import numpy as np


def count(x, new_x):
    temp = np.zeros(new_x.size, int)
    for i in range(new_x.size):
        print(new_x[i], '-', x[:, i])
        print(new_x[i] == x[:, i])
        temp[i] = np.count_nonzero(new_x[i] == x[:, i])
        print('temp[i] - ', temp[i], '\n')
    print('temp: ', temp, '\n')
    return temp.sum()


def most_possible_class(T, S, y, x_new):
    d = []
    for i in S:
        temp = T[y == i]
        d.append(temp.shape[0] * count(temp, x_new))
    d = np.array(d)
    return 'The most possible class is: ' + str(S[np.argmax(d)])


if __name__ == '__main__':
    T_test = np.array([[1, 2, 1, 3],
                       [5, 12, 7, 1],
                       [1, 2, 4, 3],
                       [1, 8, 20, 9],
                       [10, 12, 5, 1]])

    y_test = np.array([1, 2, 1, 3, 2])
    S_test = np.array([1, 2, 3])
    x_new_test = np.array([1, 12, 7, 1])

    print(most_possible_class(T_test, S_test, y_test, x_new_test))
