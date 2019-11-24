import os
import numpy as np
from oneShot import wave_reader
import concurrent.futures
import soundfile as sf
import argparse
import random
from tqdm import tqdm 


def stft_transformation(lt, input_dir, output):
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


def split_file(file_path, dir_to_new_files, lt, sec=3):
    file_name = os.path.basename(file_path).split('.')[0]
    y, sr = sf.read(file_path)
    N = sr*sec
    number_of_part = int((len(y) - N) / sr)
    for i in range(number_of_part):
        part_i = y[sr*i:(sr*i)+N]
        name = "{}_{}_{}".format(lt, file_name, i)
        stft = wave_reader.get_fft_spectrum(part_i)
        np.save(os.path.join(dir_to_new_files, name), stft)
        # for get the audio split
        #   sf.write(os.path.join(dir_to_new_files, name), part_i, 16000)


def split_all(lt, output_dir, input_dir):
    if not os.path.exists(os.path.join(output_dir, os.path.basename(input_dir))):
        os.mkdir(os.path.join(output_dir, os.path.basename(input_dir)))
    full_dir_name = os.path.join(input_dir, lt)
    my_sub_list = os.listdir(full_dir_name)
    for d in my_sub_list:
        full_file_name = os.path.join(full_dir_name, d)
        split_file(full_file_name, output_dir+'\\' + os.path.basename(input_dir), lt)


def pre_process(lt, output_dir, input_dir):
    if not os.path.exists(os.path.join(output_dir, lt)):
        os.mkdir(os.path.join(output_dir, lt))
    full_dir_name = os.path.join(input_dir, lt)
    my_sub_list = os.listdir(full_dir_name)
    for i, j in enumerate(my_sub_list):
        split_all(j, output_dir, full_dir_name)


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


def create_train_valid(output_path, input_dir, speaker_list, mode):
    train_output_dir = os.path.join(output_path, mode)
    if not os.path.exists(train_output_dir):
        os.mkdir(train_output_dir)
    print("Create {} set, please waiting..".format(mode))
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(pre_process, speaker_list, repeat(train_output_dir), repeat(input_dir))


def main(flags):
    input_dir = flags.input_woxceleb_path         # 'D:\\dataset\\woxceleb\\try_wox'
    output_path = flags.output_dir                 # 'D:\\dataset\\woxceleb\\temp_all'
    validation_percent = flags.valid_percent
    my_list = os.listdir(input_dir)
    train_speaker_number = int(len(my_list) * (1-validation_percent))
    train_speaker_list = my_list[:train_speaker_number]
    valid_speaker_list = my_list[train_speaker_number:]
    create_train_valid(output_path, input_dir, train_speaker_list, mode='train')
    create_train_valid(output_path, input_dir, valid_speaker_list, mode='valid')


if __name__ == '__main__':
    from itertools import repeat

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-woxceleb-path',
                        type=str,
                        required=True,
                        help='full path of woxceleb dataset')

    parser.add_argument('-o', '--output-dir',
                        type=str,
                        required=True,
                        help='path for the output')
    parser.add_argument('-p', '--valid-percent',
                        type=int,
                        default=0.2,
                        help='percent of validation spkear')

    FLAGS, unparsed = parser.parse_known_args()

    main(FLAGS)

    pass
