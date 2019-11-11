import os
from oneShot.model import Triplet

from oneShot.data_generator import DataGeneratorLoad

DO_TRAINING = False
model_dir_path = './save_model'
output_dir_path = './output'
weight_name = 'weights_full_model-improvement-41-0.79.hdf5'
model_name = 'model.json'
train_data_path = 'D:\\dataset\\woxceleb\\temp_train'
valid_data_path = 'D:\\dataset\\woxceleb\\temp_valid'
xvector_model = Triplet(epochs=1)
if DO_TRAINING:
    training_generator = DataGeneratorLoad(data_dir_name=train_data_path, step_per_epoch=300)
    valid_generator = DataGeneratorLoad(data_dir_name=valid_data_path, step_per_epoch=10)
    xvector_model.fit(training_generator, valid_generator, save_model=False)
else:
    xvector_model.model.load_weights(os.path.join(model_dir_path, weight_name))

# xvector_model.evaluate_model(1, train_data_path)
xvector_model.load_embedded_model()
wave_file = ".\\save_model\\merge.wav"

import numpy as np
import random


def make_oneshot(dir, N=16, dim=(512, 299), n_channels=1):
    list_dir = os.listdir(dir)
    q_batch = np.empty((N, *dim, n_channels)).astype(np.float16)
    p_batch = np.empty((N, *dim, n_channels)).astype(np.float16)
    n_batch = np.empty((N, *dim, n_channels)).astype(np.float16)
    # triple_batch = 3 * batch
    data_idx = np.random.choice(list_dir, size=N, replace=False)
    s = os.listdir(os.path.join(dir, data_idx[0]))
    choice_one_idx = random.sample(s, k=2)
    q = np.load(os.path.join(os.path.join(dir, data_idx[0]), choice_one_idx[0]))
    p = np.load(os.path.join(os.path.join(dir, data_idx[0]), choice_one_idx[1]))
    q_batch[0] = q[:, :, np.newaxis]
    p_batch[0] = p[:, :, np.newaxis]
    n_batch[0] = q[:, :, np.newaxis]   # not in used.
    for i in range(1, N):
        s = os.listdir(os.path.join(dir, data_idx[i]))
        choice_wrong_idx = random.sample(s, k=1)
        p = np.load(os.path.join(os.path.join(dir, data_idx[i]), choice_wrong_idx[0]))
        q_batch[i] = q[:, :, np.newaxis]
        p_batch[i] = p[:, :, np.newaxis]
        q_batch[i] = q[:, :, np.newaxis]  # not in used.
    triplet_batch = ({'q_input': q_batch, 'p_input': p_batch, 'n_input': q_batch},
                     {'output': np.ones(N)})
    return triplet_batch


def test_oneshot(model, n, k, dir):
    n_correct = 0
    for i in range(k):
        inputs = make_oneshot(dir, n)
        triplet_output = model.predict(inputs[0])
        triplet_output_diff = triplet_output[:, 0] - triplet_output[:, 1]
        triplet_output = np.linalg.norm(triplet_output_diff, axis=1)
        print("The predict class", np.argmin(triplet_output))
        if np.argmin(triplet_output) == 0:   # 0 is the index of the correct class
            n_correct += 1
    percent_correct = (100.0 * n_correct / k)
    print("Got an average of {}% {} way one-shot learning accuracy \n".format(percent_correct, n))
    return percent_correct


#test_oneshot(xvector_model.model, 4, 50, dir)

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


def random_result():
    result = []
    sample_number = 1000
    n_way_number = 20
    for j in range(1, n_way_number, 2):
        temp = np.random.choice(np.arange(j), sample_number)
        result.append(100*(np.sum(temp == 0) / sample_number))
    return result

#ֳֳtest_nn(10, 50, dir)
dir_train = "D:\\dataset\\woxceleb\\new_fft_train"
dir_valid = "D:\\dataset\\woxceleb\\new_fft_valid"
number_of_sample = 100
results = [[], [], []]
n_way_number = 20
for i in range(1, n_way_number, 2):
    results[0].append(test_oneshot(xvector_model.model, i, number_of_sample, dir_train))
    results[1].append(test_oneshot(xvector_model.model, i, number_of_sample, dir_valid))
    results[2].append(test_nn(i, number_of_sample, dir_train))

from matplotlib import pyplot as plt

plt.plot(np.arange(1, n_way_number, 2), results[0], label='train')
plt.plot(np.arange(1, n_way_number, 2), results[1], label='valid')
plt.plot(np.arange(1, n_way_number, 2), results[2], label='nearest neighbor')
plt.plot(np.arange(1, n_way_number, 2), results[3], label='random')
plt.plot(np.arange(1, n_way_number, 2), results[4], label='without training')
plt.xlabel("Number of possible class")
plt.ylabel("Accuracy")
plt.legend()

plt.show()

# import pickle
# with open('./outputs/NwayResults.pkl', 'wb') as f:
#     pickle.dump(results, f)






