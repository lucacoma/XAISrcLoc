import os
import matplotlib.pyplot as plt
import torch
import numpy as np
import utils
import argparse
from network_lib import EndToEndLocModel, SampleCNNLoc
from tqdm import tqdm
from params import fs, window_size,  mics,c, composite_loc_cnn, composite_sample_cnn
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
from zennit.attribution import Gradient
os.environ["CUDA_VISIBLE_DEVICES"] = '5'
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    'font.size': 15})

results_path = '/nas/home/lcomanducci/xai_src_loc/endtoend_src_loc2/results_perturbation'
parser = argparse.ArgumentParser(description='Endtoend training')
parser.add_argument('--gpu', type=str, help='gpu', default='2')
parser.add_argument('--data_path', type=str,  default='/nas/home/lcomanducci/xai_src_loc/endtoend_src_loc2/dataset2')
parser.add_argument('--T60', type=float, help='T60', default=0.6)
parser.add_argument('--SNR', type=int, help='SNR', default=10)
parser.add_argument('--log_dir',type=str, help='store tensorboard info',default='/nas/home/lcomanducci/xai_src_loc/endtoend_src_loc2/logs')
args = parser.parse_args()

SNR=args.SNR
T60 =args.T60

data_path = '/nas/home/lcomanducci/xai_src_loc/endtoend_src_loc2/dataset2/test/SNR_'+str(SNR)+'_T60_'+str(T60)
files = [os.path.join(data_path,path) for path in os.listdir(data_path)]

# LOC-CNN
saved_model_path ='/nas/home/lcomanducci/xai_src_loc/endtoend_src_loc2/models/loccnn/model_SNR_'+str(SNR)+'_T60_'+str(T60)+'.pth'
model_loc_cnn = EndToEndLocModel()
model_loc_cnn.load_state_dict(torch.load(saved_model_path))
model_loc_cnn.eval()

# SAMPLE-CNN
saved_model_path_sample_cnn='/nas/home/lcomanducci/xai_src_loc/endtoend_src_loc2/models/samplecnn/model_SNR_'+str(SNR)+'_T60_'+str(T60)+'.pth'
model_sample_cnn = SampleCNNLoc()
model_sample_cnn.load_state_dict(torch.load(saved_model_path_sample_cnn))
model_sample_cnn.eval()


time_delay_lrp_loc_cnn = []
time_delay_lrp_sample_cnn = []
time_delay_sig = []
time_delay_gt = []
N_sources = len(files)
idx_m1, idx_m2 = 5, 10 # 6,9
thresh = []
#mics_couples = [[0,1],[1,2],[2,3],[3,4],[5,6],[7,8],[9,10],[11,12],[13,14],[14,15]]
#mics_couples = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15]]

mics_couples = [[7, 8]]
for n_c in tqdm(range(len(mics_couples))):
    idx_m1, idx_m2 = mics_couples[n_c][0], mics_couples[n_c][1]
    #print('Using couple of mics {} and {}'.format(idx_m1,idx_m2))
    for n_s in tqdm(range(N_sources)):
        data_structure = np.load(str(files[n_s]))
        win_sig = data_structure['win_sig']
        sources_gt = data_structure['src_pos']
        sig_corr_time = data_structure['sig_corr_time']
        anomaly_thresh = sig_corr_time/2
        N_wins = win_sig.shape[-1]
        sources_win = []

        # LRP loc_cnn
        with Gradient(model=model_loc_cnn, composite=composite_loc_cnn) as attributor:
            out, relevance_loc_cnn = attributor(torch.permute(torch.Tensor(win_sig),(2,0,1)),
                                        torch.Tensor(data_structure['src_pos']).repeat(win_sig.shape[-1],1))

        # LRP sample_cnn
        with Gradient(model=model_sample_cnn, composite=composite_sample_cnn) as attributor:
            out, relevance_sample_cnn = attributor(torch.permute(torch.Tensor(win_sig),(2,0,1)),
                                        torch.Tensor(data_structure['src_pos']).repeat(win_sig.shape[-1],1))

        # GT GCC
        gt_sig_1 = np.zeros(window_size)
        gt_sig_2 = np.zeros(window_size)
        dist_samples_1 = int((np.linalg.norm(sources_gt - mics[:, idx_m1]) / c) * fs)
        dist_samples_2 = int((np.linalg.norm(sources_gt- mics[:, idx_m2]) / c) * fs)
        gt_sig_1[dist_samples_1] = 1 / (4 * np.pi * dist_samples_1)
        gt_sig_2[dist_samples_2] = 1 / (4 * np.pi * dist_samples_2)
        _, gcc_gt = utils.gcc_phat(gt_sig_2, gt_sig_1, fs=fs)

        dist_mic_samples = (np.linalg.norm(mics[:,idx_m1]-mics[:,idx_m2])/c) *fs

        for n_w in range(N_wins):
            # WIN SIG GCC
            _, gcc_sig = utils.gcc_phat(win_sig[idx_m2,:,n_w], win_sig[idx_m1,:,n_w], fs=fs)

            # LRP GCC - loc_cnn
            _, gcc_lrp_loc_cnn = utils.gcc_phat(relevance_loc_cnn[n_w,idx_m2,:], relevance_loc_cnn[n_w,idx_m1,:,], fs=fs)

            # LRP GCC - loc_cnn
            _, gcc_lrp_sample_cnn = utils.gcc_phat(relevance_sample_cnn[n_w,idx_m2,:], relevance_sample_cnn[n_w,idx_m1,:,], fs=fs)


            # Cycle through windows
            time_delay_lrp_loc_cnn.append(np.argmax(gcc_lrp_loc_cnn)-(len(gcc_lrp_loc_cnn)/2))
            time_delay_lrp_sample_cnn.append(np.argmax(gcc_lrp_sample_cnn)-(len(gcc_lrp_sample_cnn)/2))

            time_delay_sig.append(np.argmax(gcc_sig)-(len(gcc_sig)/2))
            time_delay_gt.append(np.argmax(gcc_gt)-(len(gcc_gt)/2))
            thresh.append(anomaly_thresh)

td_error_relevance_loc_cnn = np.abs(np.array(time_delay_gt) - np.array(time_delay_lrp_loc_cnn))
td_error_relevance_sample_cnn = np.abs(np.array(time_delay_gt) - np.array(time_delay_lrp_sample_cnn))
td_error_sig = np.abs(np.array(time_delay_gt) - np.array(time_delay_sig))

anomalies_relevance_loc_cnn = np.round(np.sum(td_error_relevance_loc_cnn>thresh)/len(td_error_relevance_loc_cnn),4)
anomalies_relevance_sample_cnn = np.round(np.sum(td_error_relevance_sample_cnn>thresh)/len(td_error_relevance_sample_cnn),4)
anomalies_sig = np.round(np.sum(td_error_sig>thresh)/len(td_error_sig),4)

MAE_sig = np.round(np.mean(td_error_sig[td_error_sig<thresh]),4)
MAE_relevance_loc_cnn = np.round(np.mean(td_error_relevance_loc_cnn[td_error_relevance_loc_cnn<thresh]),4)
MAE_relevance_sample_cnn = np.round(np.mean(td_error_relevance_sample_cnn[td_error_relevance_sample_cnn<thresh]),4)

print('Condition: SNR '+str(SNR)+' T60: '+str(T60))
print(str('MAE signal: '+str(MAE_sig)+' samples'))
print(str('MAE relevance LOC-CNN: '+str(MAE_relevance_loc_cnn)+' samples'))
print(str('MAE relevance SAMPLE-CNN: '+str(MAE_relevance_sample_cnn)+' samples'))
print(str('MAE anomalies signal: '+str(anomalies_sig)+' %'))
print(str('MAE anomalies relevance LOC-CNN: '+str(anomalies_relevance_loc_cnn)+' %'))
print(str('MAE anomalies relevance SAMPLE-CNN: '+str(anomalies_relevance_sample_cnn)+' %'))

print(str(anomalies_sig*100)+'&'+str(MAE_sig)+'&'+str(anomalies_relevance_loc_cnn*100)+'&'+str(MAE_relevance_loc_cnn)+'&'+str(anomalies_relevance_sample_cnn*100)+'&'+str(MAE_relevance_sample_cnn))
print('')

