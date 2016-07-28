from numpy import *;


def load_simple_data():
    data_matrix = matrix([
        [1., 2.1],
        [2., 1.1],
        [1.3, 1],
        [1., 1.],
        [2., 1.]
    ])
    class_labels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return data_matrix, class_labels
