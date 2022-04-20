from collections import Counter
from pprint import pprint
import numpy
from copy import deepcopy
import matplotlib.pyplot as plt


def read_traindata(filename):
    import os
    # for f in os.listdir(data_dir):
    #     input_file = os.path.join(data_dir, f)
    #     # input_file = data_dir + f"/{f}"
    #     print(input_file)
    with open(filename, "r") as fh:
        content = fh.readlines()
    return content


def write_data(new_file, content):
    with open(new_file, "w") as fh:
        fh.writelines(content)


def build_vocabulary(content, threshold):
    vocab_dict = Counter()  # defaultdict(lambda: 0)
    for i in range(len(content)):
        word_list = content[i].split()
        for j in range(1, len(word_list)):
            vocab_dict[word_list[j]] += 1

    # todo: potential bug
    vocab_arr = [word for word, count in vocab_dict.items() if count >= threshold]
    # vocab_arr = []
    # for word in vocab_dict:
    #     if vocab_dict[word] >= threshold:
    #         vocab_arr.append(word)

    return numpy.array(vocab_arr)


def get_y_train(content):
    print("Function deprecated, use build_vector()")
    return
    # y_train = []
    # for i in range(len(content)):
    #     toadd = -1 if int(content[i][0]) == 0 else int(content[i][0])
    #     y_train.append(toadd)
    # return numpy.array(y_train)


def build_vector(content, vocab):
    """
    :param content: raw email contents
    :param vocab: built vocab
    :return: x, y: x is the sample and y is the label for that sample
    """
    total_vectors = []
    y_train = []

    for i in range(len(content)):
        word_list = content[i].split()
        word_dict = dict()
        for j in range(len(word_list)):
            word_dict[word_list[j]] = 1
        vector = [0] * (len(vocab))
        for k in range(len(vocab)):
            if word_dict.get(vocab[k]):
                vector[k] = 1
        total_vectors.append(vector)
        toadd = -1 if int(content[i][0]) == 0 else int(content[i][0])
        y_train.append(toadd)

    return numpy.array(total_vectors), numpy.array(y_train)


def is_misclassified(w, x_train, y_train):
    """
    NUMPY BROADCASTING:

    w = yx + b

    if b is a scalar, and we're adding it to a vector equation, then b gets broadcasted
    Ex:

    b = 4

    w = y*x + b
    ==> b = [4, ..., 4]
    the above equation WILL work
    """
    y_pred = numpy.zeros(len(y_train))
    k = 0

    for i in range(len(x_train)):
        y_pred[i] = get_y_pred(x_train[i], w)

    for i in range(len(y_pred)):
        if y_pred[i] != y_train[i]:
            k += 1
            # optimization
            w = w + y_train[i] * x_train[i]  # w = w + yx
            # w = numpy.add(w, y_train[i] * x_train[i])  # same as the line above it
            # update rule is a big bottleneck
            # for j in range(len(x_train[0])):
            #     w[j] = w[j] + y_train[i]*x_train[i][j]
    return k, w


def train_data(x_train, y_train, max_iter=200):
    """
    1 pass through the entire dataset is called an epoch
    5 iterations --> 5 epochs
    """
    w = numpy.zeros(len(x_train[0]))
    k = 1
    iterations = 0
    while k > 0 and iterations <= max_iter:
        k, w = is_misclassified(w, x_train, y_train)
        iterations += 1
    return k, w, iterations


def get_y_pred(x_train, w):
    weighted = numpy.dot(w, x_train)
    if weighted >= 0:
        return 1
    return -1


def perceptron_test(w, test_data, y_actual):
    y_pred = numpy.zeros(len(y_actual))
    k = 0

    for i in range(len(test_data)):
        y_pred[i] = get_y_pred(test_data[i], w)
        if y_pred[i] != y_actual[i]:
            k += 1

    return k / len(y_actual)


def most_predictive(vocab, w, num_words):
    """
    w =     [w1, w2, ..., wn]
    vocab = [v1, v2, ..., vn]

    solution 0: brute force
    sort w first, then look for the top k number of words and then go about finding the words you want

    solution 1: O(nlogn) for sort
    zip them together to create a list (w_i, v_i), and then sort on w_1
    and then slice

    solution 2: argmin / argmax only return 1 element
    np.argmax(nparray) => index where the max occurs
    np.argmin(nparray) => index where min occurs

    for a sorted list of element indeces, we use np.argsort
    """
    # solution 2:
    w_sorted_index = numpy.argsort(w)
    top_positive_index = w_sorted_index[len(w_sorted_index)-num_words:][::-1]
    top_negative_index = w_sorted_index[:num_words]

    top_positive_words = [vocab[i] for i in top_positive_index]
    top_negative_words = [vocab[i] for i in top_negative_index]

    return top_positive_words, top_negative_words


def average_w(x_train, y_train, max_iter=200):
    w_all = []
    w = numpy.zeros(len(x_train[0]))
    k = 1
    iterations = 0
    while k > 0 and iterations <= max_iter:
        w_all.append(deepcopy(w))
        k, w = is_misclassified(w, x_train, y_train)
        iterations += 1

    w_all.append(w)
    w_all = numpy.array(w_all)
    w_avg = numpy.mean(w_all, axis=0)

    return k, w_avg, iterations


def evaluate_regular_percpetron(training_data, y_train, testing_data, y_validate):
    k, w, iterations = train_data(training_data, y_train, max_iter=500)
    return perceptron_test(w, testing_data, y_validate)


def evaluate_avg_percpetron(training_data, y_train, testing_data, y_validate):
    k, w, iterations = average_w(training_data, y_train, max_iter=500)
    return perceptron_test(w, testing_data, y_validate)


def run_n(content, n):
    train = content[:n]
    validate = content[n:]

    threshold = 30
    vocab = build_vocabulary(train, threshold)

    x_train, y_train = build_vector(train, vocab)
    x_validate, y_validate = build_vector(validate, vocab)

    regular_validation_err = evaluate_regular_percpetron(x_train, y_train, x_validate, y_validate)
    avg_validation_err = evaluate_avg_percpetron(x_train, y_train, x_validate, y_validate)

    return regular_validation_err, avg_validation_err


def iterations_regular_percpetron(training_data, y_train):
    k, w, iterations = train_data(training_data, y_train)
    return iterations


def iterations_avg_percpetron(training_data, y_train):
    k, w, iterations = average_w(training_data, y_train)
    return iterations


def run_n_iter(content, n):
    train = content[:n]
    # validate = content[n:]

    threshold = 30
    vocab = build_vocabulary(train, threshold)

    x_train, y_train = build_vector(train, vocab)
    # x_validate, y_validate = build_vector(validate, vocab)

    regular_iter = iterations_regular_percpetron(x_train, y_train)
    avg_iter = iterations_avg_percpetron(x_train, y_train)

    return regular_iter, avg_iter

def main():
    filename = "/Users/ldubrosa/maddie-coding/homework/machine_learning/project_1/ps1_data/spam_train.txt"
    # filename = "/Users/mo/src-control/projects/kwellerprep/privates/maddie/maddie-coding/homework/machine_learning/project_1/ps1_data/spam_train.txt"
    content = read_traindata(filename)
    train = content[:4000]
    validate = content[4000:]
    # write_data(filename2, train)
    # write_data(filename3, validate)

    # threshold = 30
    # vocab = build_vocabulary(train, threshold)
    print("done building vocab")

    # print(vocab)
    # print(len(vocab))

    # feature_vectors, y_train = build_vector(train, vocab)
    # print(feature_vectors)
    # print(feature_vectors.shape)
    # print("done building training examples")

    # print(len(feature_vectors))
    # print(feature_vectors[0])
    # print(y_train)
    # print(y_train.shape)
    # print("done building labels for training examples")

    # k, w, iterations = train_data(feature_vectors, y_train, max_iter=150)

    # feature_vectors_val, y_val = build_vector(validate, vocab)
    # print(f"done building validation set")
    # validation_error = perceptron_test(w, feature_vectors_val, y_val)
    # print(f"{validation_error*100}% incorrect")

    # num_words = 25
    # top_positive_words, top_negative_words = most_predictive(vocab, w, num_words)
    # print(f"Top Positive words: {top_positive_words}")
    # print(f"Top Negative words: {top_negative_words}")

    # import time
    # t0 = time.time()
    # k, w_avg, iterations = average_w(feature_vectors, y_train)
    # print(time.time() - t0)
    # print(w_avg)
    # print(k)
    # print(iterations)

    # feature_vectors_val, y_val = build_vector(validate, vocab)
    # print(f"done building validation set")
    # validation_error = perceptron_test(w_avg, feature_vectors_val, y_val)
    # print(f"{validation_error*100}% incorrect")

    n_arr = [100, 200, 400, 800, 2000, 4000]
    reg_plot = []
    avg_plot = []

    for n in n_arr:
        # reg_val_err, avg_val_err = run_n(content, n)
        # reg_plot.append(reg_val_err)
        # avg_plot.append(avg_val_err)
        reg_iter, avg_iter = run_n_iter(content, n)
        reg_plot.append(reg_iter)
        avg_plot.append(avg_iter)
        print(f"ran n={n}")

    plt.rcParams['figure.figsize'] = [5, 5]
    plt.scatter(n_arr, reg_plot, color="pink")
    plt.scatter(n_arr, avg_plot, color="blue")
    plt.xlabel("N")
    plt.ylabel("Validation Error")
    plt.title("Perceptron")
    plt.show()

    return

    threshold = 30
    vocab = build_vocabulary(content, threshold)
    feature_vectors, y_train = build_vector(content, vocab)
    k, w, iterations = train_data(feature_vectors, y_train, max_iter=150)

    # filename2 = "/Users/ldubrosa/maddie-coding/homework/machine_learning/project_1/ps1_data/spam_test.txt"
    filename2 = "/Users/mo/src-control/projects/kwellerprep/privates/maddie/maddie-coding/homework/machine_learning/project_1/ps1_data/spam_test.txt"
    test = read_traindata(filename2)

    feature_vectors_test, y_test = build_vector(test, vocab)
    print(f"done building test set")
    test_error = perceptron_test(w, feature_vectors_test, y_test)
    print(f"{test_error * 100}% incorrect")


main()
