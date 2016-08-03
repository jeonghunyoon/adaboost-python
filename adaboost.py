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


"""
matrix : test data
dimension : column
threshold : compare with entry of column
comparator : greater than , less than
"""


def stump_classify(data_matrix, dimension, threshold, comparator):
    return_array = ones((data_matrix.__len__(), 1))
    if comparator == 'lt':
        return_array[data_matrix[:, dimension] <= threshold] = -1
    else:
        return_array[data_matrix[:, dimension] > threshold] = -1
    return return_array


"""
data_matrix and class_labels are given,
we can get a good decision stump.
Weight_vector : we can consider weight of test data.
"""


def build_stump(data_matrix, class_labels, weight_vector):
    label_matrix = mat(class_labels).T
    m, n = shape(data_matrix)
    num_of_step = 10.0
    best_stump = {}
    best_class_estimates = mat(zeros((m, 1)))
    min_error = inf
    for i in range(n):
        range_min = data_matrix[:, i].min()
        range_max = data_matrix[:, i].max()
        step_size = (range_max - range_min) / num_of_step
        for j in range(-1, int(num_of_step) + 1):
            for comparator in ['lt', 'gt']:
                threshold = (range_min + float(j) * step_size)
                predicted_values = stump_classify(data_matrix, i, threshold, comparator)
                error_array = mat(ones((m, 1)))
                error_array[predicted_values == label_matrix] = 0
                weighted_error = weight_vector.T * error_array
                print "split: dim - %d, threshold - %.2f, comparator: %s, the weighted error - %.3f" \
                      % (i, threshold, comparator, weighted_error)
                if weighted_error < min_error:
                    min_error = weighted_error
                    best_class_estimates = predicted_values.copy()
                    best_stump['dim'] = i
                    best_stump['threshold'] = threshold
                    best_stump['comparator'] = comparator
    return best_stump, min_error, best_class_estimates


def adaboost_train(data_matrix, class_labels, num_of_iterate=40):
    weak_classifier = []
    training_data_set_size = data_matrix.__len__()
    weight_vector = mat(ones((training_data_set_size, 1)) / training_data_set_size)
    aggregated_class_estimates = mat(zeros((training_data_set_size, 1)))
    for i in range(num_of_iterate):
        best_stump, weighted_error, best_class_estimates = build_stump(data_matrix, class_labels, weight_vector)
        print "weighted_vector : ", weight_vector.T
        alpha = float(0.5 * log((1.0 - weighted_error) / max(weighted_error, 1e-16)))
        best_stump['alpha'] = alpha
        weak_classifier.append(best_stump)
        print "class estimates : ", best_class_estimates.T
        weight_vector = multiply(weight_vector, exp(multiply(-1 * alpha * mat(class_labels).T, best_class_estimates)))
        weight_vector = weight_vector / sum(weight_vector)
        aggregated_class_estimates += alpha * best_class_estimates
        print "aggregated_class_estimates : ", aggregated_class_estimates.T
        aggregated_errors = multiply(sign(aggregated_class_estimates) != mat(class_labels).T,
                                     ones((training_data_set_size, 1)))
        error_rate = aggregated_errors.sum() / training_data_set_size
        print "total error : ", error_rate, "\n"
        if error_rate == 0.0:
            break
    return weak_classifier


"""
classifier
"""


def adaboost_classifier(dataToClass, classifier_array):
    data_matrix = mat(dataToClass)
    data_size = data_matrix.__len__()
    class_estimates = mat(zeros((data_size,1)))
    for i in range(len(classifier_array)):
        class_estimate = stump_classify(data_matrix, classifier_array[i]['dim'], classifier_array[i]['threshold'], classifier_array[i]['comparator'])
        class_estimates += classifier_array[i]['alpha']*class_estimate
        print class_estimates
    return sign(class_estimates)