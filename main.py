import os
from oneShot.model import Triplet
from oneShot.data_generator import DataGeneratorLoad
from oneShot.data_process_util import make_oneshot
from matplotlib import pyplot as plt
import numpy as np
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


def compare_model():
    dir_train = "D:\\dataset\\woxceleb\\new_fft_train"
    dir_valid = "D:\\dataset\\woxceleb\\new_fft_valid"
    number_of_sample = 100
    results = [[], [], [], []]
    n_way_number = 5
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


def model_top_k():
    dir_valid = "D:\\dataset\\woxceleb\\new_fft_valid"
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
    model_dir_path = './save_model'
    output_dir_path = './output'
    weight_name = 'temp_weights-improvement-19-7.05.hdf5'
    model_name = 'model.json'
    train_data_path = 'D:\\dataset\\woxceleb\\temp_train'
    valid_data_path = 'D:\\dataset\\woxceleb\\temp_valid'
    triplet_model = Triplet(epochs=1)
    if DO_TRAINING:
        training_generator = DataGeneratorLoad(data_dir_name=train_data_path, step_per_epoch=300)
        valid_generator = DataGeneratorLoad(data_dir_name=valid_data_path, step_per_epoch=70)
        triplet_model.model.load_weights(os.path.join(model_dir_path, weight_name))
    else:
        triplet_model.model.load_weights(os.path.join(model_dir_path, weight_name))
    triplet_model.load_embedded_model()
   # results = compare_model()
    results2 = model_top_k()


    # import pickle
    # with open('./outputs/TOP_K.pkl', 'wb') as f:
    #         pickle.dump(results2, f)




