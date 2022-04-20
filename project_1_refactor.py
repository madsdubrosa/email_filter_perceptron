from collections import Counter
from pprint import pprint
import numpy
import matplotlib.pyplot as plt



def read_data(filename):
    with open(filename, "r") as fh:
        content = fh.readlines()
    return content


def build_vocabulary(content, threshold):
    vocab_dict = Counter()
    for i in range(len(content)):
        word_list = content[i].split()[1:]
        unique_words = set(word_list)
        for word in unique_words:
            vocab_dict[word] += 1

    vocab_arr = [word for word, count in vocab_dict.items() if count >= threshold]
    return numpy.array(vocab_arr)


def build_vector(content, vocab):
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


def get_y_pred(x_sample, w):
    weighted = numpy.dot(w, x_sample)
    if weighted >= 0:
        return 1
    return -1


def is_misclassified(w, x_train, y_train):
    for i in range(len(x_train)):
        y_pred = get_y_pred(x_train[i], w)
        if y_pred != y_train[i]:
            return True
    return False


def perceptron_train(x_train, y_train, max_iter=200):
    w = numpy.zeros(len(x_train[0]))
    total_mistakes = 0
    iterations = 0
    k = 0
    while is_misclassified(w, x_train, y_train) and iterations <= max_iter:
        k = 0
        for i in range(len(x_train)):
            # get prediction
            y_pred = get_y_pred(x_train[i], w)
            # compare prediction to groundtruth
            if y_pred != y_train[i]:
                total_mistakes += 1
                k += 1
                w = w + y_train[i] * x_train[i]
        iterations += 1
    k = k if iterations > max_iter else 0
    return k, w, iterations  # , total_mistakes


def perceptron_test(w, test_data, y_actual):
    y_pred = numpy.zeros(len(y_actual))
    k = 0

    for i in range(len(test_data)):
        y_pred[i] = get_y_pred(test_data[i], w)
        if y_pred[i] != y_actual[i]:
            k += 1

    return k / len(y_actual)


def most_predictive(vocab, w, num_words):
    w_sorted_index = numpy.argsort(w)
    top_positive_index = w_sorted_index[len(w_sorted_index)-num_words:][::-1]
    top_negative_index = w_sorted_index[:num_words]

    top_positive_words = [vocab[i] for i in top_positive_index]
    top_negative_words = [vocab[i] for i in top_negative_index]

    return top_positive_words, top_negative_words


def avg_perceptron_train(x_train, y_train, max_iter=200):
    w_all = numpy.zeros(len(x_train[0]))
    w = numpy.zeros(len(x_train[0]))
    total_mistakes = 0
    k = 0
    iterations = 0

    while is_misclassified(w, x_train, y_train) and iterations <= max_iter:
        k = 0
        iterations += 1
        w_all += w
        for i in range(len(x_train)):
            y_pred = get_y_pred(x_train[i], w)
            if y_pred != y_train[i]:
                total_mistakes += 1
                k += 1
                w = w + y_train[i] * x_train[i]

    w_all += w
    w_avg = w_all / iterations
    return k, w_avg, iterations


def evaluate_regular_percpetron(training_data, y_train, testing_data, y_validate):
    k, w, iterations = perceptron_train(training_data, y_train, max_iter=500)
    return perceptron_test(w, testing_data, y_validate)


def evaluate_avg_percpetron(training_data, y_train, testing_data, y_validate):
    k, w, iterations = avg_perceptron_train(training_data, y_train, max_iter=500)
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
    k, w, iterations = perceptron_train(training_data, y_train, 500)
    return iterations


def iterations_avg_percpetron(training_data, y_train):
    k, w, iterations = avg_perceptron_train(training_data, y_train, 500)
    return iterations


def run_n_iter(content, n):
    train = content[:n]

    threshold = 30
    vocab = build_vocabulary(train, threshold)

    x_train, y_train = build_vector(train, vocab)

    regular_iter = iterations_regular_percpetron(x_train, y_train)
    avg_iter = iterations_avg_percpetron(x_train, y_train)

    return regular_iter, avg_iter


def main():
    # filename = "/Users/ldubrosa/maddie-coding/homework/machine_learning/project_1/ps1_data/spam_train.txt"
    filename = "/Users/mo/src-control/projects/kwellerprep/privates/maddie/maddie-coding/homework/machine_learning/project_1/ps1_data/spam_train.txt"
    content = read_data(filename)
    train = content[:4000]
    validate = content[4000:]
    # write_data(filename2, train)
    # write_data(filename3, validate)

    # threshold = 30
    # vocab = build_vocabulary(train, threshold)
    # print("done building vocab")

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

    # k, w, iterations = perceptron_train(feature_vectors, y_train, max_iter=200)
    # print(k)
    # print(iterations)

    # feature_vectors_val, y_val = build_vector(validate, vocab)
    # print(f"done building validation set")
    # validation_error = perceptron_test(w, feature_vectors_val, y_val)
    # print(f"{validation_error*100}% incorrect")

    # num_words = 15
    # top_positive_words, top_negative_words = most_predictive(vocab, w, num_words)
    # print(f"Top Positive words: {top_positive_words}")
    # print(f"Top Negative words: {top_negative_words}")

    # k, w_avg, iterations = avg_perceptron_train(feature_vectors, y_train, 10)
    # print(w_avg)
    # print(k)
    # print(iterations)

    # feature_vectors_val, y_val = build_vector(validate, vocab)
    # print(f"done building validation set")
    # validation_error = perceptron_test(w_avg, feature_vectors_val, y_val)
    # print(f"{validation_error*100}% incorrect")

    # n_arr = [100, 200, 400, 800, 2000, 4000]
    # reg_plot = []
    # avg_plot = []
    #
    # for n in n_arr:
    #     # reg_val_err, avg_val_err = run_n(content, n)
    #     # reg_plot.append(reg_val_err)
    #     # avg_plot.append(avg_val_err)
    #     reg_iter, avg_iter = run_n_iter(content, n)
    #     reg_plot.append(reg_iter)
    #     avg_plot.append(avg_iter)
    #     print(f"ran n={n}")
    #
    # plt.rcParams['figure.figsize'] = [5, 5]
    # plt.scatter(n_arr, reg_plot, color="pink")
    # plt.scatter(n_arr, avg_plot, color="blue")
    # plt.xlabel("N")
    # plt.ylabel("Validation Error")
    # plt.title("Perceptron")
    # plt.show()
    #
    # pprint(reg_plot)

    # list_of_params_to_check = []
    # for param1 in list_of_params_to_check:
    #     for param2 in another_list_to_check:
    #         for param3...
    threshold = 50
    vocab = build_vocabulary(content, threshold)
    print(len(vocab))
    feature_vectors, y_train = build_vector(content, vocab)
    k, w, iterations = perceptron_train(feature_vectors, y_train, max_iter=100)

    # filename2 = "/Users/ldubrosa/maddie-coding/homework/machine_learning/project_1/ps1_data/spam_test.txt"
    filename2 = "/Users/mo/src-control/projects/kwellerprep/privates/maddie/maddie-coding/homework/machine_learning/project_1/ps1_data/spam_test.txt"
    test = read_data(filename2)

    feature_vectors_test, y_test = build_vector(test, vocab)
    print(f"done building test set")
    test_error = perceptron_test(w, feature_vectors_test, y_test)
    print(f"{test_error * 100}% incorrect")


main()
