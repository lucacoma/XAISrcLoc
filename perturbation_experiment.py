import os
import torch
import numpy as np
import copy
import argparse
from network_lib import EndToEndLocModel
from tqdm import tqdm
from params import  window_size, n_mic
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
from zennit.attribution import Gradient
from zennit.composites import EpsilonGammaBox

results_path = '/nas/home/lcomanducci/xai_src_loc/endtoend_src_loc2/results_perturbation'
parser = argparse.ArgumentParser(description='Endtoend training')
parser.add_argument('--gpu', type=str, help='gpu', default='0')
parser.add_argument('--T60', type=float, help='T60', default=0.15)
parser.add_argument('--SNR', type=int, help='SNR', default=25)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
SNR=args.SNR
T60 =args.T60

data_path = '/nas/home/lcomanducci/xai_src_loc/endtoend_src_loc2/dataset2/test/SNR_'+str(SNR)+'_T60_'+str(T60)
files = [os.path.join(data_path,path) for path in os.listdir(data_path)]

percentages = [0,10,20,30,40,50,60,70]

saved_model_path='/nas/home/lcomanducci/xai_src_loc/endtoend_src_loc2/models/model_SNR_'+str(SNR)+'_T60_'+str(T60)+'.pth'
model = EndToEndLocModel()
model.load_state_dict(torch.load(saved_model_path))
model = model.to(device)
model.eval()


def perturb_array(input_array,perc,mode,relevance=0):


    input_array = np.reshape(input_array,(n_mic*window_size))
    n_perc = int((perc/100)*len(input_array))

    if mode == 'relevance':
        relevance = np.reshape(relevance, (n_mic * window_size))
        idxs = np.argsort(-relevance) # order according to max indices
        idxs_zero = idxs[:n_perc]
    if mode == 'energy':
        idxs = np.argsort(-np.abs(input_array)) # order according to max indices
        idxs_zero = idxs[:n_perc]
    if mode =='random':
        idxs_zero = np.random.choice(np.arange(len(input_array)),n_perc,replace=False)

    perturbed_array=copy.deepcopy(input_array)
    for i in range(len(idxs_zero)):
        perturbed_array[idxs_zero[i]]=0

    perturbed_array = np.reshape(perturbed_array,(n_mic,window_size))
    return perturbed_array


MAE_relevance = np.zeros(len(percentages))
MAE_random = np.zeros(len(percentages))
MAE_energy = np.zeros(len(percentages))

import time

for p in tqdm(range(len(percentages))):
    sources_est_relevance = []
    sources_est_random = []
    sources_est_energy = []
    sources_gt = []
    N_sources = len(files)

    for n_s in tqdm(range(N_sources)):

        data_structure = np.load(str(files[n_s]))
        win_sig = data_structure['win_sig']
        N_wins = win_sig.shape[-1]

        composite = EpsilonGammaBox(low=win_sig.min(), high=win_sig.max())
        with Gradient(model=model, composite=composite) as attributor:
            out, relevance = attributor(torch.permute(torch.Tensor(win_sig),(2,0,1)).to(device),
                                        torch.Tensor(data_structure['src_pos']).repeat(win_sig.shape[-1],1).to(device))

        relevance = relevance.cpu().numpy()
        tic = time.time()
        sig_perturb_relevance = np.zeros((N_wins,n_mic,window_size))
        sig_perturb_energy =np.zeros((N_wins,n_mic,window_size))
        sig_perturb_random =np.zeros((N_wins,n_mic,window_size))
        for n_w in range(N_wins):
            input_win_sig = win_sig[:, :, n_w]
            relevance_win= relevance[n_w]

            sig_perturb_relevance[n_w] = perturb_array(input_win_sig, percentages[p], mode='relevance', relevance=relevance_win)
            sig_perturb_energy[n_w] = perturb_array(input_win_sig, percentages[p], mode='energy')
            sig_perturb_random[n_w] = perturb_array(input_win_sig, percentages[p], mode='random')

        with torch.no_grad():
            est_pos_relevance = model(torch.Tensor(sig_perturb_relevance).to(device)).cpu().numpy()
            est_pos_energy = model(torch.Tensor(sig_perturb_energy).to(device)).cpu().numpy()
            est_pos_random = model(torch.Tensor(sig_perturb_random).to(device)).cpu().numpy()

        for n_w in range(N_wins):
            sources_est_relevance.append(est_pos_relevance[n_w])
            sources_est_energy.append(est_pos_energy[n_w])
            sources_est_random.append(est_pos_random[n_w])
            sources_gt.append(data_structure['src_pos'])

    MAE_relevance[p] = np.mean(np.abs(np.array(sources_gt) - np.array(sources_est_relevance)))
    MAE_random[p] = np.mean(np.abs(np.array(sources_gt) - np.array(sources_est_random)))
    MAE_energy[p] = np.mean(np.abs(np.array(sources_gt) - np.array(sources_est_energy)))

results_split_path = os.path.join(results_path, 'SNR_' + str(SNR) + '_T60_' + str(T60))
if not os.path.exists(results_path):
    os.makedirs(results_path)
if not os.path.exists(results_split_path):
    os.makedirs(results_split_path)

np.save(os.path.join(results_split_path,'relevance.npy'),MAE_relevance)
np.save(os.path.join(results_split_path,'energy.npy'),MAE_energy)
np.save(os.path.join(results_split_path,'random.npy'),MAE_random)




