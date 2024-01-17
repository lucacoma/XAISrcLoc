import argparse
import os
import torch.nn
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import network_lib
import datetime
from params import window_size
parser = argparse.ArgumentParser(description='Endtoend training')
parser.add_argument('--gpu', type=str, help='gpu', default='2')
parser.add_argument('--data_path', type=str,  default='/nas/home/lcomanducci/xai_src_loc/endtoend_src_loc2/dataset2')
parser.add_argument('--T60', type=float, help='T60', default=0.6)
parser.add_argument('--SNR', type=int, help='SNR', default=10)
parser.add_argument('--log_dir',type=str, help='store tensorboard info',default='/nas/home/lcomanducci/xai_src_loc/endtoend_src_loc2/logs')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
window_size = 1280
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
T60 = args.T60
SNR = args.SNR





class EndToEndDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, window_size):
        self.data_path = data_path
        self.files = [os.path.join(self.data_path,path) for path in os.listdir(self.data_path)]
        self.window_size = window_size
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Load data structure

        data_structure = np.load(str(self.files[idx]))

        # Load windowed signal

        win_sig = data_structure['win_sig']

        N_wins = win_sig.shape[-1]
        idx_slice = torch.randint(low=0, high=N_wins,size=(1,))
        # N.B. transpose is due to channel first pytorch convention
        win_sig_tensor = torch.from_numpy(win_sig)[:,:,idx_slice].squeeze(-1)
        win_sig_tensor = torch.Tensor(win_sig_tensor.detach().numpy())


        # Load source position
        src_pos = data_structure['src_pos']
        src_pos = torch.Tensor(src_pos)


        return win_sig_tensor, src_pos

def train_epoch(train_dataloader, model, device,loss_fn,optimizer):

    num_batches = len(train_dataloader.dataset)
    running_loss = 0.

    for batch, (win_sig_batch, src_loc_batch) in enumerate(train_dataloader):
        win_sig_batch, src_loc_batch = win_sig_batch.to(device), src_loc_batch.to(device)

        optimizer.zero_grad(set_to_none=True)

        src_loc_batch_est = model(win_sig_batch)

        # Loss and backprop
        loss = loss_fn(src_loc_batch_est,src_loc_batch)
        loss.backward()
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
    running_loss/=num_batches
    return running_loss

def val_epoch(val_dataloader, model, device,loss_fn):

    num_batches = len(val_dataloader.dataset)
    running_loss = 0.

    with torch.no_grad():
        for batch, (win_sig_batch, src_loc_batch) in enumerate(val_dataloader):
            win_sig_batch, src_loc_batch = win_sig_batch.to(device), src_loc_batch.to(device)
            src_loc_batch_est = model(win_sig_batch)

            # Loss and backprop
            loss = loss_fn(src_loc_batch_est,src_loc_batch)

            # Gather data and report
            running_loss += loss.item()
    running_loss/=num_batches
    return running_loss

def main():
    saved_model_path='/nas/home/lcomanducci/xai_src_loc/endtoend_src_loc2/models/model'+'_SNR_'+str(SNR)+'_T60_'+str(T60)+'.pth'
    model = network_lib.EndToEndLocModel()
    model = model.to(device)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 1000
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.2,patience=100,verbose=1)

    log_name = os.path.join(args.log_dir,'SNR_'+str(SNR)+'_T60_'+str(T60)+'_'+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    if not os.path.exists(log_name):
        os.makedirs(log_name)

    writer = SummaryWriter(log_dir=log_name)

    train_path = os.path.join(args.data_path,'train','SNR_'+str(SNR)+'_T60_'+str(T60))
    val_path =  os.path.join(args.data_path,'val','SNR_'+str(SNR)+'_T60_'+str(T60))

    training_data = EndToEndDataset(train_path,window_size)
    val_data = EndToEndDataset(val_path,window_size)
    batch_size = 100

    train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True,num_workers=4)
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True,num_workers=4)

    #win_sig, src_loc = next(iter(train_dataloader))

    model = model.cuda()
    early_stop_patience = 200
    for n_e in tqdm(range(epochs)):
        model.train(True)
        train_loss = train_epoch(train_dataloader, model, device,loss_fn,optimizer)
        model.eval()
        val_loss = val_epoch(val_dataloader, model, device,loss_fn)
        scheduler.step(val_loss)
        # Write to tensorboard
        writer.add_scalar('Loss/train', train_loss, n_e)
        writer.add_scalar('Loss/val', val_loss, n_e)
        writer.flush()
        # Early Stopping and best checkpoint model
        # Handle saving best model + early stopping
        if n_e == 0:
            val_loss_best = val_loss
            early_stop_counter = 0
            saved_model_path = saved_model_path

            torch.save(model.state_dict(), saved_model_path)
        if n_e > 0 and val_loss < val_loss_best:
            saved_model_path = saved_model_path
            torch.save(model.state_dict(), saved_model_path)
            val_loss_best = val_loss
            # print(f'Model saved epoch{n_e}')
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print('Patience status: ' + str(early_stop_counter) + '/' + str(early_stop_patience))

        # Early stopping
        if early_stop_counter > early_stop_patience:
            print('Training finished at epoch ' + str(n_e))
            break


if __name__=='__main__':
    main()