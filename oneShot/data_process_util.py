import os
import numpy as np
from oneShot import wave_reader
import concurrent.futures
import soundfile as sf
import shutil
import random


output_path = 'D:\\dataset\\woxceleb\\temp_yaniv'
input_dir_path = 'D:\\dataset\\woxceleb\\train_split'


def stft_transformation(lt, input_dir=input_dir_path, output=output_path):
    os.mkdir(os.path.join(output, lt))
    print(lt)
    full_dir_name = os.path.join(input_dir, lt)
    my_sub_list = os.listdir(full_dir_name)
    count = 0
    for j in my_sub_list:
        full_file_name = os.path.join(full_dir_name, j)
        stft = wave_reader.get_fft_spectrum(full_file_name)
        np.save(output+'\\' + lt + '\\'+str(count), stft)
        count += 1


def split_file(file_path, dir_to_new_files, sec=3):
    file_name = os.path.basename(file_path).split('.')[0]
    y, sr = sf.read(file_path)
    N = sr*sec
    number_of_part = int((len(y) - N) / sr)
    for i in range(number_of_part):
        part_i = y[sr*i:(sr*i)+N]
        name = file_name + '_' + str(i) + '.wav'
        sf.write(os.path.join(dir_to_new_files, name), part_i, 16000)


def split_all(lt, output_dir=output_path, input_dir='D:\\dataset\\woxceleb\\yaniv_input'):
    os.mkdir(os.path.join(output_dir, lt))
    print(lt)
    full_dir_name = os.path.join(input_dir, lt)
    my_sub_list = os.listdir(full_dir_name)
    for d in my_sub_list:
        full_file_name = os.path.join(full_dir_name, d)
        split_file(full_file_name, output_dir+'\\' + lt)


def reduce_dir(lt, output_dir=output_path, input_dir='D:\\dataset\\woxceleb\\yaniv_input'):
    os.mkdir(os.path.join(output_dir, lt))
    full_dir_name = os.path.join(input_dir, lt)
    my_sub_list = os.listdir(full_dir_name)
    count = 0
    for i, j in enumerate(my_sub_list):
        full_file_name = os.path.join(full_dir_name, j)
        my_sub_file = os.listdir(full_file_name)
        for file_ in my_sub_file:
            full_file_name_path = os.path.join(full_file_name, file_)
            print(full_file_name_path)
            shutil.copy(full_file_name_path, output_dir+'\\' + lt + '\\'+str(count)+'.wav')
            count += 1


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

if __name__ == '__main__':

    # input_dir = 'D:\\dataset\\woxceleb\\train_split'
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     dir_list = os.listdir(input_dir)
    #     dir_list = dir_list[:10]
    #     executor.map(stft_transformation, dir_list)    # methods to execute calls asynchronously
    # reduce_dir('id10001')
    # input_dir = 'D:\\dataset\\woxceleb\\yaniv_input'
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     my_list = os.listdir(input_dir)
    #     executor.map(reduce_dir, my_list)
    # split_all('id10001')
    # input_dir = 'D:\\dataset\\woxceleb\\yaniv_input'
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     my_list = os.listdir(input_dir)
    #     executor.map(split_all, my_list)
    pass
