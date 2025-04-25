# twdlstm test v0.5

import sys # CLI arguments: print(sys.argv)
import os # os.getcwd, os.chdir
from datetime import datetime # datetime.now
from zoneinfo import ZoneInfo # ZoneInfo in datetime.now
import yaml # yaml.safe_load
import time # wallclock = time.time()
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import r2_score, median_absolute_error
from collections import OrderedDict # saving/loading model state_dict

# torch.cuda.is_available() # check GPU

#%% read config yaml
# os.chdir('/mydata/forestcast/william/WP3') # setwd()

path_config = str(sys.argv[1])
# print(path_config)

with open(path_config) as cf_file:
    config = yaml.safe_load(cf_file.read())

# print(config)


#%% general setup
colvec = [
	'#ff0f0f','#1b0fff','#009e1d','#f5d000', # red/blue/green/yellow
	'#ff1abe','#00a4d1','#77cc00','#ff950a', # pink/sky/chartr/orange
	'#9c00eb','#00e2e6','#00c763','#bd6b00'  # violet/cyan/algae/brown
]
# plt.scatter(range(len(colvec)),range(len(colvec)),color=colvec)
# plt.savefig('plot.pdf')
# plt.close()


#%% load tstoy data, loop over series and stack
path_tstoy = config['path_data'] + '/tstoy' + config['tstoy'] + '/'
# '/mydata/forestcast/william/WP3/DataProcessed/tstoy04/'

# now = datetime.now() # UTC by def on runai
now = datetime.now(tz=ZoneInfo("Europe/Zurich"))
now_str = now.strftime("%Y-%m-%d %H:%M:%S")
print(now_str + ' running twdlstm test v0.5\n')
# print('\n')

print('Supplied config:')
print(path_config+'\n')
# print('\n')

# seriesvec = config['series'] # up to v0.3.2
# seriesvec = config['series_trva'] # as of v0.4, distinguish from series_te
seriesvec = config['series_te'] # as of v0.4
covvec = config['covvec']

nT = config['nT'] # time window length
# ^ taking same time window as in tr

nb_series = len(seriesvec)
nb_cov = len(covvec)

# ini: s=0
s = 0
series_s = seriesvec[s] # '01' # 01-42
path_csv_series_s = (path_tstoy + 'SeparateSeries/tstoy' + config['tstoy'] +
    '_series_' + series_s + '.csv'
)
dat_s = pd.read_csv(
    path_csv_series_s,
    header=0,
    # nrows=2 # to check cols
    dtype={
        'ts':str,
        'twd':float,
        'pr':float,
        'at':float,
        'ws':float,
        'dp':float,
        'sr':float,
        'lr':float,
        'vp':float
    }
)

ind_t0 = int(np.where(dat_s['ts']==config['date_t0'])[0].item())
# ^ dat row index corresponding to date_t0 in config

ind_t = range(ind_t0, nT+ind_t0, 1) # for i in ind_t: print(i)

y_full = dat_s['twd'][ind_t]
x_full = dat_s[covvec].iloc[ind_t,:] # ini, time subset, all cols
# y_full.shape
# x_full.shape

# v0.4: use entire time window (te) for cov norm TODO: check
mean_s = np.apply_along_axis(np.mean, 0, x_full) # tr cov mean
sd_s = np.apply_along_axis(np.std, 0, x_full) # tr cov sd
for j in range(nb_cov): # loop over columns, overwrite each cov
    x_full.iloc[:,j] = (x_full.iloc[:,j]-mean_s[j])/sd_s[j]

x_full = np.expand_dims(x_full, axis=0) # add a dim for stacking series
y_full = np.expand_dims(y_full, axis=0) # add a dim for stacking series

for s in range(1,nb_series): # loop over series after 1st 
    series_s = seriesvec[s] # '01' # 01-42
    path_csv_series_s = (path_tstoy + 'SeparateSeries/tstoy' + config['tstoy'] +
        '_series_' + series_s + '.csv'
    )
    dat_s = pd.read_csv(
        path_csv_series_s,
        header=0,
        dtype={
            'ts':str,
            'twd':float,
            'pr':float,
            'at':float,
            'ws':float,
            'dp':float,
            'sr':float,
            'lr':float,
            'vp':float
        }
    )
    y_s_full = dat_s['twd'][ind_t]
    x_s_full = dat_s[covvec].iloc[ind_t,:]
    mean_s = np.apply_along_axis(np.mean, 0, x_s_full)
    sd_s = np.apply_along_axis(np.std, 0, x_s_full)
    for j in range(nb_cov): # loop over columns, overwrite each cov
        x_s_full.iloc[:,j] = (x_s_full.iloc[:,j]-mean_s[j])/sd_s[j]
    
    x_s_full = np.expand_dims(x_s_full, axis=0) # add a dim for stacking series
    y_s_full = np.expand_dims(y_s_full, axis=0) # add a dim for stacking series
    x_full = np.concatenate((x_full, x_s_full), axis=0)
    y_full = np.concatenate((y_full, y_s_full), axis=0)

# end loop over s in seriesvec

# x_full.shape
# y_full.shape


#%% torch tensor (check CPU or GPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# print(device)
torch.set_default_device(device)


xfull = torch.tensor(x_full, dtype=torch.float32)
yfull = torch.tensor(y_full, dtype=torch.float32)
# xfull.shape # # series as dim 0
# yfull.shape # series as dim 0


#%% "batches": index of start of overlaping temporal subsets, for each series
b_len = int(config['batch_len'])
b_nb = int(nT - b_len + 1) # int(nT_tr - b_len + 1)
# ^ b_nb = number of temporal batches per series, each of length b_len

# nb_batches = int(nb_series*b_nb)

# xb = torch.empty(size=(nb_batches, b_len, nb_cov))
# yb = torch.empty(size=(nb_batches, b_len))
# for s in range(nb_series): # loop over series (dim 0)
#     x_s = torch.select(xfull, dim=0, index=s)
#     y_s = torch.select(yfull, dim=0, index=s) # .reshape(-1,1)
#     for t in range(b_nb):
#         ind_t = range(t, int(b_len+t)) # overlapping temporal subsets
#         xb[t+s*b_nb,:,:] = x_s[ind_t,:]
#         yb[t+s*b_nb,:] = y_s[ind_t]

# # xb.shape # batches = dim 0
# # yb.shape # batches = dim 0

# hor = config['loss_hor']
hor = 1
# ^ v0.4.2: hor fixed to 1, only last obs of each batch contributes to loss
ind_hor = range(int(b_len-hor),b_len)
slice_hor = int(b_len-hor) # assuming hor=1
# ^ indices of obs contributing to loss eval within each batch



#%% create tr and va subsets of batches
# nb_va = int(np.floor(nb_batches*config['prop_va']))
# # ^ number of batches for va, out of nb_batches
# nb_tr = nb_batches - nb_va
# # ^ number of batches for tr
# # nb_va/nb_batches # should be close to config['prop_va']

# # np.random.seed(seed=int(config['srs_seed'])) # fix seed simple random sampling
# # np.random.choice(range(5), size=2, replace=False) # global seed fixed
# rng = np.random.default_rng(seed=int(config['srs_seed'])) # local rand generator
# ind_va = np.sort(rng.choice(range(nb_batches), size=nb_va, replace=False))
# ind_tr = np.array(list(set(range(nb_batches)).difference(ind_va)))
# # ^ set diff: in range but not in ind_va

# print('Indices of va batches:\n',', '.join(map(str, ind_va)),'\n')

# nb_tr_loss = nb_tr*hor
# # ^ number of contributions to tr loss, some obs counted more than once if hor>1
# #   because batches overlap in time (shifted by 1 time step from one another)
# nb_va_loss = nb_va*hor



#%% LSTM  model class
i_size = nb_cov # xb.shape[2] # nb cols in x = nb input features 
h_size = config['h_size']
o_size = config['o_size']
nb_layers = config['nb_layers']

class Model_LSTM(torch.nn.Module):
    def __init__(self, input_size, d_hidden, num_layers, output_size):
        super().__init__()
        self.d_hidden = d_hidden
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=d_hidden,
            num_layers=num_layers,
            batch_first=True
        )
        # self.drop = torch.nn.Dropout(p=0.5)
        self.linear = torch.nn.Linear(
            in_features=d_hidden,
            out_features=output_size
        )
        self.actout = torch.nn.ReLU()
    
    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = self.get_hidden(x)
        x, hidden = self.lstm(x, hidden)
        # x = self.actout(self.linear(self.drop(x)))
        x = self.actout(self.linear(x))
        return x, hidden
    
    def get_hidden(self, x):
        # second axis = batch size, i.e. x.shape[0] when batch_first=True
        hidden = (
            torch.zeros(
                self.num_layers,
                x.shape[0],
                self.d_hidden,
                device=x.device
            ),
            torch.zeros(
                self.num_layers,
                x.shape[0],
                self.d_hidden,
                device=x.device
            )
        )
        return hidden


model = Model_LSTM(i_size, h_size, nb_layers, o_size) # instantiate
# model.train() # print(model)


#%% initial values
path_ckpt = config['path_checkpointdir'] + '/' + config['prefixoutput'] + '_ckpt_best.pt'
# ^ by default takes "best" = smallest va loss in epochs

model.load_state_dict(torch.load(path_ckpt, weights_only=False)) # checkpoint
# ^ <All keys matched successfully> = ok

h0 = torch.zeros(nb_layers, h_size) # num_layers, hidden_size
c0 = torch.zeros(nb_layers, h_size) # num_layers, hidden_size
# ^ as in tr/va

nb_param = 4*h_size*i_size + 4*h_size*h_size + 4*h_size*2 + o_size*(h_size+1)
nb_obs = len(seriesvec)*nT # 
print('Number of parameters =',nb_param)
print('Total number of te observations =',nb_obs,'\n')
# print('\n')


#%% pred over all "batches" over all series, extract corresponding obs
model.eval()

# assuming hor=1, so one obs per batch
yte = np.zeros((nT,nb_series))
yte_pred = np.zeros((nT,nb_series))
# ^ rows are time points (incl ini), cols are series

if device.type=='cuda': # need to transfer from GPU to CPU for np
    for s in range(nb_series): # loop over series (dim 0)
        x_s = torch.select(xfull, dim=0, index=s)
        # y_s = torch.select(yfull, dim=0, index=s)
        yte[:,s] = torch.select(yfull, dim=0, index=s).cpu().detach().numpy()       
        for t in range(b_nb):
            ind_t = range(t, int(b_len+t)) # overlapping temporal subsets
            fwdpass_st = model(x_s[ind_t,:], (h0,c0)) # from ini
            yte_pred[ind_t[slice_hor],s] = fwdpass_st[0][ind_hor].cpu().detach().numpy().item()
            # yte[ind_t[slice_hor],s] = y_s[ind_t][ind_hor].reshape(-1,1).cpu().detach().numpy().item()
else: # then device.type='cpu'
    for s in range(nb_series): # loop over series (dim 0)
        x_s = torch.select(xfull, dim=0, index=s)
        # y_s = torch.select(yfull, dim=0, index=s)
        yte[:,s] = torch.select(yfull, dim=0, index=s).detach().numpy()
        for t in range(b_nb):
            ind_t = range(t, int(b_len+t)) # overlapping temporal subsets
            fwdpass_st = model(x_s[ind_t,:], (h0,c0)) # from ini
            yte_pred[ind_t[slice_hor],s] = fwdpass_st[0][ind_hor].detach().numpy().item()
            # yte[ind_t[slice_hor],s] = y_s[ind_t][ind_hor].reshape(-1,1).detach().numpy().item()


#%% compute te perf metric by series
ind_t_pred = range(int(b_len-hor), nT) # burn-in of b_len-1

biasvec = np.zeros(nb_series)
scalevec = np.zeros(nb_series)
rvec = np.zeros(nb_series)
r2vec = np.zeros(nb_series)
MedAEvec = np.zeros(nb_series)
for s in range(nb_series): # loop over series (dim 0)
    biasvec[s] = np.mean(yte[ind_t_pred,s]) - np.mean(yte_pred[ind_t_pred,s])
    scalevec[s] = np.std(yte[ind_t_pred,s])/np.std(yte_pred[ind_t_pred,s])
    rvec[s] = np.corrcoef(yte[ind_t_pred,s], yte_pred[ind_t_pred,s])[0,1]
    r2vec[s] = r2_score(yte[ind_t_pred,s], yte_pred[ind_t_pred,s])
    MedAEvec[s] = median_absolute_error(yte[ind_t_pred,s], yte_pred[ind_t_pred,s])
    print('* Series',seriesvec[s],'te metrics:') # 
    print('  - mean(obs)-mean(pred) =',round(biasvec[s],4)) # diff of means
    print('  - sd(obs)/sd(pred) =',round(scalevec[s],4)) # ratio of scales
    print('  - lin corr =',round(rvec[s],4)) # lin corr
    print('  - R^2 =',round(r2vec[s],4)) # R^2 on te batches, by series
    print('  - MedAE =',round(MedAEvec[s],4)) # R^2 on te batches, by series


#%% plot pred
path_out_te = config['path_outputdir'] + '/' + config['prefixoutput'] + '_te'
if not os.path.exists(path_out_te):
    os.makedirs(path_out_te) # create te output dir if does not exist

for s in range(nb_series): # loop over series (dim 0)
    plt.figure(figsize=(12,6))
    plt.scatter(range(nT), yte[:,s], s=10, c='grey', label='ini') # s=16
    plt.scatter(ind_t_pred, yte[ind_t_pred,s], s=16, c=colvec[1], label='te')
    plt.plot(ind_t_pred, yte_pred[ind_t_pred,s],
        linewidth=1, c='black', label='pred')
    plt.legend(loc='upper left')
    plt.title('series ' + seriesvec[s])
    plt.savefig(path_out_te + '/ts_series' + seriesvec[s] + '.pdf')
    plt.close()


print('\n')
now_again_str = now.strftime("%Y-%m-%d %H:%M:%S")
print(now_again_str)
print('done')

# END twdlstm test
