import pyroomacoustics as pra
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

windows = [1280, 2560, 5120]
window_size = 5120
if window_size == 5120:
    fc_size = 3712
if window_size == 1280:
    fc_size = 896

n_mic = 16
fs = 16000
c = 343

# Add microphones to 3D room
height_mic = 1.2
mic_center = np.array([1.7, 7.0, 0.96])
R = pra.linear_2D_array(mic_center[:2], M=n_mic, phi=0, d=0.15)
R = np.concatenate((R, np.ones((1, n_mic)) * height_mic), axis=0)
mics = pra.MicrophoneArray(R, fs).R

SEED =5000
height_mic = 1.2
nx, ny = 65, 65
X, Y = np.meshgrid(np.linspace(mics[0,:].min()+0.25, mics[0,:].max()-0.25, nx), np.linspace(4.5, 6.5, ny))

src_pos = np.array((np.ravel(X), np.ravel(Y))).T
rng = np.random.default_rng(seed=SEED)
height_srcs = rng.uniform(low=1, high=1.5,size=(src_pos.shape[0],))
src_pos = np.concatenate((src_pos, np.expand_dims(height_srcs,axis=-1)),axis=-1)
max_src, min_src = np.max(src_pos), np.min(src_pos)

src_pos_norm = (src_pos-min_src)/(max_src-min_src)
src_pos_norm = -1 +(2*src_pos_norm)

SEED = 5000

x_min = 1.45
x_max = 1.95
y_min = 5.25
y_max = 5.75
src_pos_test = []
idx_delete = []
for n_s in range(len(src_pos)):
    if src_pos[n_s,0]> x_min and src_pos[n_s,0]<x_max and src_pos[n_s,1] > y_min and src_pos[n_s,1] < y_max:
        src_pos_test.append(src_pos[n_s])
        idx_delete.append(n_s)
src_pos = np.delete(src_pos,idx_delete,axis=0)
src_pos_test = np.array(src_pos_test)
src_pos_train, src_pos_val = train_test_split(src_pos, test_size=0.2, random_state=SEED)

PLOT_SETUP=False
if PLOT_SETUP:
    plt.figure()
    plt.plot(mics[0,:],mics[1,:],'b*')
    plt.plot(src_pos_train[:,0],src_pos_train[:,1],'b*')
    plt.plot(src_pos_val[:,0],src_pos_val[:,1],'r*')
    plt.plot(src_pos_test[:,0],src_pos_test[:,1],'g*')
    plt.axis('equal')
    plt.show()

# audio signals
corpus = pra.datasets.CMUArcticCorpus(download=True)
idx_tracks = rng.choice(np.arange(len(corpus)),size=src_pos_train.shape[0]+src_pos_val.shape[0]+src_pos_test.shape[0])
idx_tracks_train=idx_tracks[:src_pos_train.shape[0]]
idx_tracks_val=idx_tracks[src_pos_train.shape[0]:src_pos_train.shape[0]+src_pos_val.shape[0]]
idx_tracks_test=idx_tracks[src_pos_train.shape[0]+src_pos_val.shape[0]:src_pos_train.shape[0]+src_pos_val.shape[0]+src_pos_test.shape[0]]

