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


def pegasos_svm_train(x_train, y_train, pegasos_lambda, max_iterations=20):
    w = numpy.zeros(len(x_train[0]))
    t = 0

    for iteration in range(0, max_iterations):
        for j in range(len(x_train)):
            t = t+1
            stepsize = 1/(t * pegasos_lambda)
            test_value = y_train[j]*(numpy.dot(w, x_train[j]))
            if test_value < 1:
                w = ((1-(stepsize*pegasos_lambda))*w) + (stepsize*y_train[j]*x_train[j])
            else:
                w = ((1-(stepsize*pegasos_lambda))*w)

    return w


def hinge_loss(w, x, y):
    return max(0, 1-(y*numpy.dot(w, x)))


def evaluate_pegasos(x_data, y_data, pegasos_lambda, w):
    """
    evaluate the SVM objective
    :param x_data: x values in the split
    :param y_data:  y values in the split
    :param pegasos_lambda: lambda
    :param w: weight vector
    :return: f(w) = (lambda/2) * ||w||^2 + (1/len(x_data)) * SUM_all { hinge_loss(x, y, w) }
    """
    running_sum = 0
    for i in range(len(y_data)):
        running_sum += hinge_loss(w, x_data[i], y_data[i])

    return ((pegasos_lambda / 2) * (numpy.linalg.norm(w)**2)) + ((1/len(x_data))*running_sum)


def run_pegasos_train(x_train, y_train, pegasos_lambda, iteration_range):
    p_plot = []

    for iteration in range(1, iteration_range + 1):
        w_new = pegasos_svm_train(x_train, y_train, pegasos_lambda, iteration)
        eval_num = evaluate_pegasos(x_train, y_train, pegasos_lambda, w_new)
        p_plot.append(eval_num)
        print(f"ran {iteration}")

    return numpy.array(p_plot)

def pegasos_svm_test(x_data, y_data, pegasos_lamdba, w):
    """
    evaluation metric function for determining accuracy
    y_i * (w.dot(x)) < 0
    :param x_train:
    :param y_train:
    :param pegasos_lamdba:
    :param w:
    :return:
    """
    k = 0
    for i in range(len(x_data)):
        y_pred = numpy.dot(w, x_data[i])
        val_check = y_data[i] * y_pred
        if val_check < 0:
            k += 1
    return k / len(x_data)


def run_train_diff_lambdas(x_train, y_train, max_iterations, pegasos_lambdas):
    accuracy_plot = []
    hinge_loss_plot = []
    w_plot = []
    for pegasos_lambda in pegasos_lambdas:
        w = pegasos_svm_train(x_train, y_train, pegasos_lambda, max_iterations)
        w_plot.append(w)
        accuracy = pegasos_svm_test(x_train, y_train, pegasos_lambda, w)
        running_sum = 0
        for i in range(len(x_train)):
            running_sum += hinge_loss(w, x_train[i], y_train[i])
        avg_hinge_loss = running_sum / len(x_train)
        accuracy_plot.append(accuracy)
        hinge_loss_plot.append(avg_hinge_loss)
    return accuracy_plot, hinge_loss_plot, w_plot

def val_diff_lambdas(x_val, y_val, max_iterations, pegasos_lambdas, w_plot):
    accuracy_plot = []
    for i in range(len(pegasos_lambdas)):
        accuracy = pegasos_svm_test(x_val, y_val, pegasos_lambdas[i], w_plot[i])
        accuracy_plot.append(accuracy)
    return accuracy_plot

def run_diff_lambdas(x_train, y_train, x_val, y_val, max_iterations, pegasos_lambdas):
    train_error, hinge_loss_plot, w_plot = run_train_diff_lambdas(x_train, y_train, max_iterations, pegasos_lambdas)
    val_error = val_diff_lambdas(x_val, y_val, max_iterations, pegasos_lambdas, w_plot)
    return train_error, hinge_loss_plot, val_error


def find_support_vectors(x_data, y_data, w):
    k = 0
    for x_sample, y_sample in zip(x_data, y_data):
        hl = hinge_loss(w, x_sample, y_sample)
        if hl != 0:
            k += 1
    return k

def multi_class(k, x_train, y_train, pegasos_lambda, max_iterations=20):
    all_w = []
    for w in range(k):
        all_w.append(numpy.zeros(len(x_train[0])))
    all_w = numpy.array(all_w)
    t = 0

    for iteration in range(0, max_iterations):
        for j in range(len(x_train)):
            t = t + 1
            stepsize = 1 / (t * pegasos_lambda)
            for k in range(len(all_w)):
                w = all_w[k]
                test_value = y_train[j] * (numpy.dot(w, x_train[j]))
                if y_train[j] > 0 and test_value < 1 and k < len(all_w)-1:
                    all_w[k] = ((1 - (stepsize * pegasos_lambda)) * w) + (stepsize * y_train[j] * x_train[j])
                elif y_train[j] > 0 and k < len(all_w)-1:
                    all_w[k] = ((1 - (stepsize * pegasos_lambda)) * w)
                elif y_train[j] < 0 and test_value < 1:
                    all_w[k] = ((1 - (stepsize * pegasos_lambda)) * w) + (stepsize * y_train[j] * x_train[j])
                else:
                    all_w[k] = ((1 - (stepsize * pegasos_lambda)) * w)

    return all_w

def pegasos_multiclass_test(x_data, y_data, all_w):
    k = 0
    for i in range(len(x_data)):
        y_pred = numpy.argmax(numpy.dot(all_w, x_data[i]))
        val_check = y_data[i] * y_pred
        if val_check < 0:
            k += 1
    return k / len(x_data)

def main():
    filename = "spam_train.txt"
    content = read_data(filename)
    train = content[:4000]
    validate = content[4000:]

    threshold = 30
    vocab = build_vocabulary(train, threshold)
    print("done building vocab")

    x_train, y_train = build_vector(train, vocab)
    print("done building training examples")

    x_val, y_val = build_vector(validate, vocab)
    print("done building validating examples")

    pegasos_lambda = 2**-5

    # t_arr = [i * len(x_train) for i in range(1, 21)]
    # p_plot = run_pegasos_train(x_train, y_train, pegasos_lambda, 20)
    # print(t_arr[19], p_plot[19])
    # print(len(p_plot))
    #
    # plt.rcParams['figure.figsize'] = [5, 5]
    # plt.scatter(t_arr, p_plot, color="pink")
    # plt.xlabel("T")
    # plt.ylabel("P")
    # plt.title("pegasos")
    # plt.show()

    # w = pegasos_svm_train(x_train, y_train, pegasos_lambda, 20)
    # accuracy = pegasos_svm_test(x_val, y_val, pegasos_lambda, w)
    # print(accuracy)


    # pegasos_lambdas = [2**i for i in range(-9, 2)]
    # train_error, hinge_loss_plot, val_error = run_diff_lambdas(x_train, y_train, x_val, y_val, 20, pegasos_lambdas)
    #
    # plt.rcParams['figure.figsize'] = [5, 5]
    # plt.scatter(pegasos_lambdas, train_error, color="pink")
    # plt.scatter(pegasos_lambdas, hinge_loss_plot, color="blue")
    # plt.scatter(pegasos_lambdas, val_error, color="red")
    # plt.xlabel("lambda")
    # plt.ylabel("error")
    # plt.title("pegasos")
    # plt.show()
    #
    # min_val_error = min(val_error)
    # min_val_err_lambda = pegasos_lambdas[numpy.argmin(val_error)]
    # print(min_val_error, min_val_err_lambda)

    # pegasos_lambdas = [2**i for i in range(-9, 2)]
    # train_error, hinge_loss_plot, val_error = run_diff_lambdas(x_train, y_train, x_val, y_val, 20, pegasos_lambdas)


    # pegasos_lambdas = [numpy.log2(pl) for pl in pegasos_lambdas]
    # log_pl = list(range(-9, 2))
    # plt.rcParams['figure.figsize'] = [5, 5]
    # plt.scatter(log_pl, train_error, color="pink")
    # plt.scatter(log_pl, hinge_loss_plot, color="blue")
    # plt.scatter(log_pl, val_error, color="red")
    # plt.xlabel("lambda")
    # plt.ylabel("error")
    # plt.title("pegasos")
    # plt.show()

    # min_val_error = min(val_error)
    # min_val_err_lambda = pegasos_lambdas[numpy.argmin(val_error)]
    # print(min_val_error, min_val_err_lambda)


    # filename2 = "spam_test.txt"
    # test = read_data(filename2)
    #
    # threshold = 30
    # vocab = build_vocabulary(content, threshold)
    # print("done building vocab")
    #
    # x_train, y_train = build_vector(content, vocab)
    # print("done building training examples")
    #
    # x_test, y_test = build_vector(test, vocab)
    # print("done building test examples")
    #
    # w = pegasos_svm_train(x_train, y_train, 2**-8, 20)
    # test_accuracy = pegasos_svm_test(x_test, y_test, pegasos_lambda, w)
    # print(test_accuracy)
    #
    # train_k = find_support_vectors(x_train, y_train, w)
    # print(f"number of support vectors = {train_k}")

    all_w = multi_class(2, x_train, y_train, pegasos_lambda, 20)
    print("dont training multiclass")

    val_accuracy = pegasos_multiclass_test(x_val, y_val, all_w)
    print(f"accuracy = {val_accuracy}")


main()

