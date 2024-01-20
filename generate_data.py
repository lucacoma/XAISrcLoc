import numpy as np
import pyroomacoustics as pra
import tensorflow as tf
import argparse
import os
import gpuRIR
import utils
gpuRIR.activateMixedPrecision(False)
gpuRIR.activateLUT(True)
from tqdm import tqdm
from params import window_size,corpus, idx_tracks_train,idx_tracks_val, src_pos_train, src_pos_val,src_pos_test,idx_tracks_test
from params import mics, n_mic

parser = argparse.ArgumentParser(description='Endtoend data generation')
parser.add_argument('--T60', type=float, help='T60', default=0.1)
parser.add_argument('--SNR', type=int, help='SNR', default=40)
parser.add_argument('--gpu', type=str, help='gpu', default='0')
path = '/nas/home/lcomanducci/xai_src_loc/endtoend_src_loc2/dataset2'
args = parser.parse_args()
T60 = args.T60
SNR = args.SNR
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import torch



# Specify room dimensions
room_dim = [3.6, 8.2, 2.4]  # meters
e_absorption, max_order = pra.inverse_sabine(T60, room_dim)

for data_split in ['train','val','test']:
    print('Computing '+str(data_split) + ' data')

    if data_split == 'train':
        sources_pos = src_pos_train
        corpus_idxs = idx_tracks_train
    if data_split == 'val':
        sources_pos = src_pos_val
        corpus_idxs = idx_tracks_val
    if data_split == 'test':
        sources_pos = src_pos_test
        corpus_idxs = idx_tracks_test

    for j in tqdm(range(len(sources_pos))):
        signal = corpus[corpus_idxs[j]].data
        fs = corpus[corpus_idxs[j]].fs

        # Convert signal to float
        signal = signal / (np.max(np.abs(signal)))

        # Compute Signal Correlation Time
        sig_corr_time = utils.compute_correlation_time(signal)


        # Add source to 3D room (set max_order to a low value for a quick, but less accurate, RIR)
        source_position = sources_pos[j]
        # Add microphones to 3D room


        att_diff = 15.0  # Attenuation when start using the diffuse reverberation model [dB]
        att_max = 60.0  # Attenuation at the end of the simulation [dB]
        fs = 16000.0  # Sampling frequency [Hz]

        beta = gpuRIR.beta_SabineEstimation(room_dim, T60)  # Reflection coefficients
        Tmax = T60
        nb_img = gpuRIR.t2n(Tmax, room_dim)  # Number of image sources in each dimension
        RIRs = gpuRIR.simulateRIR(room_dim, beta, np.expand_dims(source_position,1).T, mics.T, nb_img, Tmax, fs)[0]

        fft_len = len(signal) +RIRs.shape[1] -1
        SIG = torch.fft.fft(torch.Tensor(signal),n=fft_len)
        RIRs_fft = torch.fft.fft(torch.tensor(RIRs),n=fft_len, dim=1)
        signal_conv = torch.fft.ifft(torch.multiply(SIG, RIRs_fft),dim=1)

        # AWGN
        noisy_signal_conv, noise = utils.add_white_gaussian_noise(signal_conv.detach().numpy(), SNR)
        noisy_signal_conv = torch.Tensor(noisy_signal_conv)

        # Split in windows
        N_wins = int(noisy_signal_conv.shape[-1]/window_size)
        frames = torch.reshape(noisy_signal_conv[:,:N_wins*window_size],(n_mic,N_wins,window_size))
        win_sig  = torch.permute(frames, (0,2,1))

        # Save data
        if data_split =='train' or data_split == 'val':
            train_path = os.path.join(path,data_split)
            train_split_path = os.path.join(train_path, 'SNR_' + str(SNR) + '_T60_' + str(T60))
            if not os.path.exists(train_path):
                os.makedirs(train_path)
            if not os.path.exists(train_split_path):
                os.makedirs(train_split_path)
            np.savez(file=os.path.join(train_split_path,str(j)), signal=noisy_signal_conv,
                     src_pos=source_position,
                     win_sig=win_sig)

        if data_split =='test':
            test_path = os.path.join(path,'test')
            test_split_path = os.path.join(test_path, 'SNR_' + str(SNR) + '_T60_' + str(T60))
            if not os.path.exists(test_path):
                os.makedirs(test_path)
            if not os.path.exists(test_split_path):
                os.makedirs(test_split_path)
            np.savez(file=os.path.join(test_split_path, str(j)), signal=noisy_signal_conv,
                     src_pos=source_position,
                     win_sig=win_sig,
                     sig_corr_time=sig_corr_time)



