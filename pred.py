# twdlstm pred v0.6.2

import sys # CLI arguments: print(sys.argv)
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

# path_config = '/mydata/forestcast/william/WP3/LSTM_preds/configs/config_71-01.yaml'
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
# '/mydata/forestcast/william/WP3/DataProcessed/tstoy05/'

# now = datetime.now() # UTC by def on runai
now = datetime.now(tz=ZoneInfo("Europe/Zurich"))
now_str = now.strftime("%Y-%m-%d %H:%M:%S")
print(now_str + ' running twdlstm pred v0.6.2\n')
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

# print(z0[:,50,150])

# names_cov = ['pr', 'at', 'ws', 'dp', 'sr', 'lr']
names_cov = ['pr', 'at', 'ws', 'dp', 'sr', 'lr', 'vp', 'sw'] # updated 2025-06-19
# ^ all cov processed on grid covering CH, ordering matters

seriesvec = config['series_trva'] # different "ensemble member" for each CV fold
covvec = config['covvec'][:] # "[:]" to copy, not reference
# nT = config['nT'] # time window length, to be split in batches for tr/va

nb_series = len(seriesvec)
# nb_cov = len(covvec)

# dy not in gridded cov
if 'dy' in covvec: # if daily cov is in covvec
    dy_in_cov = True # daily cov is in covvec
    ind_dy = covvec.index('dy') # index of daily cov
    covvec.remove('dy') # remove from covvec, add back after
else:
    dy_in_cov = False # daily cov is not in covvec

# print(covvec)
nb_cov = len(covvec) # nb of cov, excl dy

ind_cov = [] # ini empty
for j in covvec:
    ind_cov.append(names_cov.index(j)) # append index of kept cov in order

x_full = z0[ind_cov,:,:] # ini at actual pred day = lag 0

# normalize each cov by its respective mean/sd (over grid points)
mean_s = x_full.mean(axis=(1, 2)) # cov mean
sd_s = x_full.std(axis=(1, 2))    # cov sd
for j in range(nb_cov): # loop over columns, overwrite each cov
    x_full[j,:,:] = (x_full[j,:,:]-mean_s[j])/sd_s[j]


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
    zt = zt[ind_cov,:,:]
    # normalize each cov by its respective mean/sd (over grid points) 
    mean_s = zt.mean(axis=(1, 2)) # cov mean
    sd_s = zt.std(axis=(1, 2))    # cov sd
    for j in range(nb_cov): # loop over columns, overwrite each cov
        zt[j,:,:] = (zt[j,:,:]-mean_s[j])/sd_s[j]
    
    zt = np.expand_dims(zt, axis=0) # add a dim for stacking series
    # x_full = np.concatenate((x_full,zt), axis=0) # stack after => wrong
    x_full = np.concatenate((zt,x_full), axis=0) # stack before => good
    # ^ stack before because t lag goes backwards in time

# x_full.shape # daily lags stacked in dim 0, keep only necessary cov in dim 1


# add dy if in covvec
if dy_in_cov:
    path_dy_csv = path_tstoy + 'cov/tstoy' + config['tstoy'] + '_dy.csv'
    dat_dy = pd.read_csv(
        path_dy_csv,
        header=0,
        dtype={'ts':str, 'dy':float}
    )
    dy_day_pred = dat_dy['doy'][dat_dy['ts']==config['pred_day']].iloc[0]
    cov_dy = np.arange(dy_day_pred-b_len,dy_day_pred+1)
    # len(cov_dy) # x_full.shape[0]
    
    cov_dy = (cov_dy-np.mean(cov_dy))/np.std(cov_dy) # normalize
    
    cov_dy = np.reshape(np.repeat(cov_dy, x_full.shape[2]*x_full.shape[3]),
        (cov_dy.shape[0],x_full.shape[2],x_full.shape[3]))
    # ^ repeat cov_dy for each x/y coord, then reshape to grid dimensions
    cov_dy = np.expand_dims(cov_dy, axis=1) # add a dim for stacking cov
    # cov_dy.shape
    
    if ind_dy == x_full.shape[1]:
        # if dy is last cov in x_full
        x_full = np.append(x_full,cov_dy,axis=1)
    else:
        # if dy is not last cov in x_full
        # x_full = np.insert(x_full, 1, cov_dy, axis=1) # not tested
        print('\n','dy not last cov in covec,code not tested at l. 165, quitting')
        quit()
    
    covvec = config['covvec'] # restore covvec with dy
    nb_cov = len(covvec) # nb of cov, incl dy

# x_full.shape
# x_full[10,:,155,200]
# x_full[10,:,155,210]
# x_full[12,:,155,210]
# ^ check that dy does not vary across x/y coords but varies in time


#%% torch tensor (check CPU or GPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# print(device)
torch.set_default_device(device)

xfull = torch.tensor(x_full, dtype=torch.float32)

# xfull.shape


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
h0 = torch.zeros(nb_layers, h_size, device=device) # num_layers, hidden_size
c0 = torch.zeros(nb_layers, h_size, device=device) # num_layers, hidden_size

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




#%% load best/last set of param among epochs (best = smallest va loss)
path_ckpt = config['path_checkpointdir'] + '/' + config['prefixoutput']

ind_hor = -1 # last obs

if config['source']=='cv':
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
elif config['source']=='train':
    ypred = np.zeros(shape=(1, x_full.shape[2], x_full.shape[3]))
    wallclock0 = time.time()
    if config['whichckpt']=='best':
        path_best_ckpt = path_ckpt+'_ckpt_best.pt'
    elif config['whichckpt']=='last':
        path_best_ckpt = path_ckpt+'_ckpt_'+str(config['maxepoch'])+'.pt'
    
    model.load_state_dict(torch.load(path_best_ckpt, weights_only=False))
    # ^ <All keys matched successfully> = ok
    model.eval() #
    for xc in range(x_full.shape[2]):
        # xc = 0
        for yc in range(x_full.shape[3]):
            # yc = 0
            fwdpass = model(xfull[:,:,xc,yc], (h0,c0))
            ypred[0,xc,yc] = fwdpass[0][ind_hor]
    
    wallclock1 = time.time() # in seconds
    print('double for loop over x/y took',round((wallclock1 - wallclock0)/60,1),'m\n')


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
