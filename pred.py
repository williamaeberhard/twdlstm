# twdlstm pred v0.5.1

import sys # CLI argumennts: print(sys.argv)
import os # os.getcwd, os.chdir
from datetime import datetime # datetime.now # datetime class
import datetime as dtm # dtm module
from zoneinfo import ZoneInfo # ZoneInfo in datetime.now
import yaml # yaml.safe_load
import time # wallclock = time.time()
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
# from sklearn.metrics import r2_score, median_absolute_error
from collections import OrderedDict # saving/loading model state_dict
import zarr

#%% read config yaml
# os.chdir('/mydata/forestcast/william/WP3') # setwd()

path_config = str(sys.argv[1])
# path_config = '/mydata/forestcast/william/WP3/LSTM_preds/configs/config_17-00.yaml'
# path_config = '/mydata/forestcast/william/WP3/LSTM_preds/configs/config_31-00.yaml'
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
# '/mydata/forestcast/william/WP3/DataProcessed/tstoy05/'

# now = datetime.now() # UTC by def on runai
now = datetime.now(tz=ZoneInfo("Europe/Zurich"))
now_str = now.strftime("%Y-%m-%d %H:%M:%S")
print(now_str + ' running twdlstm pred v0.5.1\n')
# print('\n')

print('Supplied config:')
print(path_config+'\n')
# print('\n')


#%% read zarr daily cov grid, create array for all lags
path_covgrid = config['path_dailycovgrid']

day_pred = ''.join(config['pred_day'].split('-'))
# ^ 'YYYYMMDD' format

z0 = zarr.load(path_covgrid+'/'+day_pred+'.zarr')
# z0.shape # 6 cov, 320 x coords, 224 y coords
# ^ ini with day to predict on

names_cov = ['pr', 'at', 'ws', 'dp', 'sr', 'lr']
# ^ all cov processed on grid covering CH, ordering matters

seriesvec = config['series_trva'] # different "ensemble member" for each CV fold
covvec = config['covvec']
# nT = config['nT'] # time window length, to be split in batches for tr/va

ind_cov = [] # ini empty
for j in covvec:
    ind_cov.append(names_cov.index(j)) # append index of kept cov in order

x_full = z0[ind_cov,:,:] # ini at actual pred day = lag 0
x_full = np.expand_dims(x_full, axis=0) # add a dim for stacking series
# x_full.shape # keep only necessary cov. dim 0 = daily lags

b_len = int(config['batch_len'])
# ^ nb of cov grid zarr to load, eahc lagged by 1 day wrt day_pred

day_0 = datetime.strptime(day_pred,'%Y%m%d')
oneday = dtm.timedelta(days=1) # one unit of temporal lag

for t in range(1,b_len+1): # lag 1:b_len
    # t = 1
    day_t = day_0 - t*oneday # lag one day for each t
    dayt = day_t.strftime('%Y%m%d') # 'YYYYMMDD' format
    
    zt = zarr.load(path_covgrid+'/'+dayt+'.zarr')
    zt = np.expand_dims(zt[ind_cov,:,:], axis=0) # add a dim for stacking series
    # zt.shape # 6 cov, 320 x coords, 224 y coords
    
    x_full = np.concatenate((x_full,zt), axis=0)

# x_full.shape # daily lags stacked in dim 0, keep only necessary cov in dim 1


#%% torch tensor (check CPU or GPU)
nb_series = len(seriesvec)
nb_cov = len(covvec)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# print(device)
torch.set_default_device(device)

xfull = torch.tensor(x_full, dtype=torch.float32)
# yfull = torch.tensor(y_full, dtype=torch.float32)

# xfull.shape
# yfull.shape


#%% LSTM model class
i_size = nb_cov # xb.shape[2] # nb cols in x = nb input features 
h_size = config['h_size']
o_size = config['o_size']
nb_layers = config['nb_layers']

if config['actout']=='ReLU':
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
elif config['actout']=='Softplus':
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
            self.actout = torch.nn.Softplus()
        
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
elif config['actout']=='Sigmoid':
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
            self.actout = torch.nn.Sigmoid()
        
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
h0 = torch.zeros(nb_layers, h_size) # num_layers, hidden_size
c0 = torch.zeros(nb_layers, h_size) # num_layers, hidden_size

# # torch.manual_seed(config['seed'])
# tgen = torch.Generator(device=device).manual_seed(config['torch_seed'])

# state_dict_inirand = OrderedDict({
#     'lstm.weight_ih_l0': torch.randn(4*h_size,i_size,device=device,generator=tgen),
#     'lstm.weight_hh_l0': torch.randn(4*h_size,h_size,device=device,generator=tgen),
#     'lstm.bias_ih_l0': torch.randn(4*h_size,device=device,generator=tgen),
#     'lstm.bias_hh_l0': torch.randn(4*h_size,device=device,generator=tgen),
#     'linear.weight': torch.randn(o_size,h_size,device=device,generator=tgen),
#     'linear.bias': torch.randn(o_size,device=device,generator=tgen)
# })
# # print(state_dict_inirand['linear.bias'])
# # print(model.state_dict()['linear.bias'])


# #%% construct Laplacian regularization matrix
# len_reg = config['len_reg']
# # ^ nb of pred at end of batch that are Laplacian-regularized
# lapmat = np.eye(len_reg,k=-1)+np.eye(len_reg,k=1)-2*np.eye(len_reg)
# lapmat[0,0] = -1.0                     # rows should add up to 0
# lapmat[len_reg-1,len_reg-1] = -1.0 # rows should add up to 0
# # ^ 1st order differences, to apply on y_pred for every batch
# lap = torch.tensor(lapmat, dtype=torch.float32)
# hp_lambda = config['lambda_LaplacianReg']/len_reg




#%% for fold i: load best set of param among epochs (best = smallest va loss)
path_ckpt = config['path_checkpointdir'] + '/' + config['prefixoutput']

ind_hor = -1 # last obs

ypred = np.zeros(shape=(nb_series, x_full.shape[2], x_full.shape[3]))

wallclock0 = time.time()
for i in range(nb_series): # i index identifies held-out series
    # i = 0
    print('--- fold',i,': held-out series =',seriesvec[i])
    path_best_ckpt = path_ckpt+'_ckpt_best_fold'+str(i)+'.pt'
    
    model.load_state_dict(torch.load(path_best_ckpt, weights_only=False))
    # ^ <All keys matched successfully> = ok
    model.eval() #
    
    #%% for fold i: forward pass for pred, loop over x/y coords
    for xc in range(x_full.shape[2]):
        # xc = 0
        for yc in range(x_full.shape[3]):
            # yc = 0
            fwdpass = model(xfull[:,:,xc,yc], (h0,c0))
            ypred[i,xc,yc] = fwdpass[0][ind_hor]

# end for loop over i index for LOO-CV folds ("ensemble members")
wallclock1 = time.time() # in seconds
print('triple for loop over folds/x/y took',round((wallclock1 - wallclock0)/60,1),'m\n')


#%% write pred array into a zarr file
# z_ypred = zarr.array(ypred)
# z_ypred.shape # dim 0 = ensemble members, then x/y coords

path_out_pred = config['path_outputdir'] + '/' + config['prefixoutput']
path_out_pred = path_out_pred + '-' + config['pred_run']

if not os.path.exists(path_out_pred):
    os.makedirs(path_out_pred) # create te output dir if does not exist

# zarr.save(path_out_pred+'/pred.zarr',ypred) # bad with Rarr::read_zarr_array
zarr.save_array(
	store=path_out_pred+'/pred.zarr',
	arr=ypred,
	zarr_format=2
) # good with Rarr::read_zarr_array


print('\n')
nowagain = datetime.now(tz=ZoneInfo("Europe/Zurich"))
now_again_str = nowagain.strftime("%Y-%m-%d %H:%M:%S")
print(now_again_str)
print('done')

# END twdlstm pred
