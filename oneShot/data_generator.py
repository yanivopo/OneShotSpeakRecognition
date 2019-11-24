# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 22:41:52 2019

@author: USER
"""
import numpy as np
import keras
import random
import os
import pickle
import argparse
from tqdm import tqdm


class DataGeneratorCreate(keras.utils.Sequence):
    def __init__(self, dir_name, batch_size=16, dim=(512, 299), n_channels=1, step_per_epoch=5000,
                 output_dir='./batch_train/'):
        self.dim = dim
        self.batch_size = batch_size
        self.dir_name = dir_name
        self.n_channels = n_channels
        self.output_dir = output_dir
        self.step_per_epoch = step_per_epoch

    def __batch_generation(self, arr_list, index):
        # Initialization
        q_batch = np.empty((self.batch_size, *self.dim, self.n_channels)).astype(np.float16)
        p_batch = np.empty((self.batch_size, *self.dim, self.n_channels)).astype(np.float16)
        n_batch = np.empty((self.batch_size, *self.dim, self.n_channels)).astype(np.float16)
        #triple_batch = 3 * batch
        data_idx = np.random.choice(arr_list, size=(self.batch_size, 2))
        full_dir_name_one_idx = np.core.defchararray.add(self.dir_name + '\\', data_idx[:, 0])
        full_dir_name_sec_idx = np.core.defchararray.add(self.dir_name + '\\', data_idx[:, 1])

        for i in range(self.batch_size):
            file_one_idx = os.listdir(full_dir_name_one_idx[i])
            file_sec_idx = os.listdir(full_dir_name_sec_idx[i])
            choice_one_idx = random.sample(file_one_idx, k=2)
            choice_sec_idx = random.sample(file_sec_idx, k=1)
            q = np.load(os.path.join(full_dir_name_one_idx[i], choice_one_idx[0]))
            p = np.load(os.path.join(full_dir_name_one_idx[i], choice_one_idx[1]))
            n = np.load(os.path.join(full_dir_name_sec_idx[i], choice_sec_idx[0]))
            q_batch[i] = q[:, :, np.newaxis]
            p_batch[i] = p[:, :, np.newaxis]
            n_batch[i] = n[:, :, np.newaxis]
        triplet_batch = ({'q_input': q_batch, 'p_input': p_batch, 'n_input': n_batch},
                         {'output': np.ones(self.batch_size)})
        with open(self.output_dir + str(index) + '.pkl', 'wb') as handle:
            pickle.dump(triplet_batch, handle)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.step_per_epoch

    def __getitem__(self, index):
        'Generate one batch of data'
        my_list = os.listdir(self.dir_name)
        arr_list = np.array(my_list)
        # Generate batch
        self.__batch_generation(arr_list, index)


class DataGeneratorLoad(keras.utils.Sequence):
    def __init__(self, step_per_epoch, data_dir_name):
        self.step_per_epoch = step_per_epoch
        self.data_dir_name = data_dir_name

    def __len__(self):
        return self.step_per_epoch

    def __getitem__(self, index):
        # load batch
        batch = self.__data_generation_load()
        return batch

    def __data_generation_load(self, ):
        my_list = os.listdir(self.data_dir_name)
        rand_index = np.random.choice(len(my_list), 1)[0]
        with open(os.path.join(self.data_dir_name, my_list[rand_index]), 'rb') as handle:
            batch_load = pickle.load(handle)
        return batch_load


def create_triplet(dir_input, mode, number_of_batch, batch_size):
    print("create {} triplet".format(mode))
    dir_full_input = os.path.join(dir_input, mode)
    dir_output = os.path.join(dir_input, 'triplet_{}\\'.format(mode))
    if not os.path.exists(dir_output):
        os.mkdir(dir_output)
    generator_create = DataGeneratorCreate(dir_full_input, step_per_epoch=number_of_batch,
                                           output_dir=dir_output, batch_size=batch_size)
    for _ in tqdm(generator_create):
        pass


def main(flags):
    dir_input = flags.input_dataProcess_path
    batch_size = flags.batch_size
    number_of_batch_train = flags.number_of_batch_train
    number_of_batch_valid = flags.number_of_batch_valid
    mode = 'train'

    create_triplet(dir_input, mode, number_of_batch_train, batch_size)
    mode = 'valid'
    create_triplet(dir_input, mode, number_of_batch_valid, batch_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-dataProcess-path',
                        type=str,
                        required=True,
                        help='full path of woxceleb dataset')

    parser.add_argument('-b', '--batch-size',
                        type=int,
                        help='The number of batch size')

    parser.add_argument('-nt', '--number-of-batch-train',
                        type=int,
                        default=5000,
                        help='The number of different batch for the train')

    parser.add_argument('-nv', '--number-of-batch-valid',
                        type=int,
                        default=500,
                        help='The number of different batch for the valid')

    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS)
