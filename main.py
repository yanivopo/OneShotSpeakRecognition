import os
from oneShot.model import Triplet
from oneShot.data_generator import DataGeneratorLoad
from data_process_util import make_oneshot
from matplotlib import pyplot as plt
import numpy as np
import argparse
DO_TRAINING = False


def test_nn(n, k, dir):
    n_correct = 0
    for i in range(k):
        inputs = make_oneshot(dir, n)
        query = inputs[0]['q_input'].astype(np.float64)
        positive = inputs[0]['p_input'].astype(np.float64)
        diff_q_p = query - positive
        diff_q_p = diff_q_p.reshape(diff_q_p.shape[0], -1)
        nn_output = np.linalg.norm(diff_q_p, axis=1)
        print("The predict class", np.argmin(nn_output))
        if np.argmin(nn_output) == 0:   # 0 is the index of the correct class
            n_correct += 1
    percent_correct = (100.0 * n_correct / k)
    print("Got an average of {}% {} way nearest neighbor accuracy \n".format(percent_correct, n))
    return percent_correct


def random_result(min_n_way=1, n_way_number=20):
    result = []
    sample_number = 1000

    for j in range(min_n_way, n_way_number, 2):
        temp = np.random.choice(np.arange(j), sample_number)
        result.append(100*(np.sum(temp == 0) / sample_number))
    return result


def compare_model(pre_process_path):
    dir_train = os.path.join(pre_process_path, 'train')
    dir_valid = os.path.join(pre_process_path, 'valid')
    number_of_sample = 100
    results = [[], [], [], []]
    n_way_number = 9
    min_n_way = 3
    for i in range(min_n_way, n_way_number, 2):
        results[0].append(triplet_model.evaluate_model(dir_train, i, number_of_sample))
        results[1].append(triplet_model.evaluate_model(dir_valid, i, number_of_sample))
        results[2].append(test_nn(i, number_of_sample, dir_train))
    results[3].extend(random_result(min_n_way, n_way_number))
    plt.plot(np.arange(min_n_way, n_way_number, 2), results[0], label='train')
    plt.plot(np.arange(min_n_way, n_way_number, 2), results[1], label='valid')
    plt.plot(np.arange(min_n_way, n_way_number, 2), results[2], label='nearest neighbor')
    plt.plot(np.arange(min_n_way, n_way_number, 2), results[3], label='random')
    plt.xlabel("Number of possible class")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
    return results


def model_top_k(pre_process_path):
    dir_valid = os.path.join(pre_process_path, 'valid')
    number_of_sample = 200
    n_way_number = 15
    results_k = []
    for k, j in enumerate(range(n_way_number, 21)):
        results_k.append([])
        for i in range(1, 6):
            results_k[k].append(triplet_model.evaluate_model(dir_valid, j, number_of_sample, top_of_k=i))
    # plt.plot(np.arange(1, 6), results_k, label='n_way:{}'.format(n_way_number))
    # plt.xlabel("Number of Top K")
    # plt.ylabel("Accuracy")
    # plt.legend()
    # plt.show()
    return results_k


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-od', '--output-dir',
                        type=str,
                        default='./output',
                        help='output path')

    parser.add_argument('-wn', '--weight-name',
                        type=str,
                        default='weights_full_model-improvement-41-0.79.hdf5',
                        help='weights file name')

    parser.add_argument('-tdp', '--train-data-path',
                        type=str,
                        required=True,
                        help='The output path of preProcess')

    parser.add_argument('-m', '--mode',
                        choices=['train', 'load'],
                        default='load',
                        help='Whether to train new model or load existing model')

    parser.add_argument('-te', '--run_test_evaluate',
                        choices=['True', 'False'],
                        default='True',
                        help='Whether to run evaluate method to compare the algorithm ')

    FLAGS, unparsed = parser.parse_known_args()

    triplet_model = Triplet(epochs=1)
    if FLAGS.mode == 'train':
        training_generator = DataGeneratorLoad\
            (data_dir_name=os.path.join(FLAGS.train_data_path, "triplet_train"), step_per_epoch=100)
        valid_generator = \
            DataGeneratorLoad(data_dir_name=os.path.join(FLAGS.train_data_path, "triplet_valid"), step_per_epoch=70)
        history = triplet_model.fit(training_generator, valid_generator, save_model=True,
                                    save_model_dir=FLAGS.model_dir_path)
    else:
        triplet_model.model.load_weights(os.path.join('./save_model', FLAGS.weight_name))
    triplet_model.load_embedded_model()
    if FLAGS.run_test_evaluate == 'True':
        results = compare_model(FLAGS.train_data_path)
        results2 = model_top_k(FLAGS.train_data_path)
