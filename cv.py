# twdlstm cv v0.6.4

import sys # CLI argumennts: print(sys.argv)
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

#%% read config yaml
# os.chdir('/mydata/forestcast/william/WP3') # setwd()

# path_config = '/mydata/forestcast/william/WP3/LSTM_runs/configs/config_00.yaml'
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
print(now_str + ' running twdlstm cv v0.6.4\n')
# print('\n')

print('Supplied config:')
print(path_config+'\n')
# print('\n')

# seriesvec = config['series'] # up to v0.3.2
seriesvec = config['series_trva'] # as of v0.4, distinguish from series_te
covvec = config['covvec']
nT = config['nT'] # time window length, to be split in batches for tr/va

nb_series = len(seriesvec)
nb_cov = len(covvec)

# Load and stack all series (as before)
s = 0
series_s = seriesvec[s]
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
        'vp':float,
        'sw':float,
        'dy':float,
        'el':float
    }
)

ind_t0 = int(np.where(dat_s['ts']==config['date_t0'])[0].item())
ind_t = range(ind_t0, nT+ind_t0, 1)

y_full = dat_s['twd'][ind_t]
x_full = dat_s[covvec].iloc[ind_t,:]
x_full = np.expand_dims(x_full, axis=0)
y_full = np.expand_dims(y_full, axis=0)

for s in range(1,nb_series):
    series_s = seriesvec[s]
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
            'vp':float,
            'sw':float,
            'dy':float,
            'el':float
        }
    )
    y_s_full = dat_s['twd'][ind_t]
    x_s_full = dat_s[covvec].iloc[ind_t,:]
    x_s_full = np.expand_dims(x_s_full, axis=0)
    y_s_full = np.expand_dims(y_s_full, axis=0)
    x_full = np.concatenate((x_full, x_s_full), axis=0)
    y_full = np.concatenate((y_full, y_s_full), axis=0)

# Normalize features after stacking all series
# x_full shape: (nb_series, nT, nb_cov)
mean_all = np.mean(x_full, axis=(0,1))  # mean over all series and time
std_all = np.std(x_full, axis=(0,1))
x_full = (x_full - mean_all) / std_all

# x_full.shape
# y_full.shape


#%% check if some series are fully nan, stop CV script if yes
stopscript = False
serivesfullnan = [] # ini empty
for i in range(nb_series):
    if np.all(np.isnan(y_full[i,:])):
        stopscript = True
        # serivesfullnan.append(i)
        serivesfullnan.append(seriesvec[i])

# serivesfullnan

if stopscript:
    print('Series',serivesfullnan,'full of nan, stoppping.')
    quit()



#%% torch tensor (check CPU or GPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# print(device)
torch.set_default_device(device)

xfull = torch.tensor(x_full, dtype=torch.float32)
yfull = torch.tensor(y_full, dtype=torch.float32)

# xfull.shape
# yfull.shape



#%% setup static input features (z)
zvec = config['zvec']
z_size = len(zvec) # size of z vector = nb static input features
z_fc_size = int(config['z_fc_size']) # size of z vector

path_csv_static = (
    path_tstoy + 'tstoy' + config['tstoy'] + '_staticcov.csv'
)
dat_static = pd.read_csv(
    path_csv_static,
    header=0,
    dtype={
        'id':str,   # id_sisp
        'ea':float, # mch_easting
        'no':float, # mch_northing
        'el':float  # mch_elevation
    }
)

zmat = dat_static[dat_static['id'].isin(seriesvec)] # subset series
zmat = zmat[zvec] # keep user-supplied static features

mean_z = np.mean(zmat, axis=0) # mean over series
std_z = np.std(zmat, axis=0) # sd over series
zmat = (zmat - mean_z) / std_z # normalize static features, overwrite

# zmat.shape # nb_series, z_size

zb = torch.tensor(zmat.values, dtype=torch.float32, device=device)
# zb.shape # nb_series, z_size



#%% LSTM model class
i_size = nb_cov # xb.shape[2] # nb cols in x = nb input features 
h_size = config['h_size']
o_size = config['o_size']
nb_layers = config['nb_layers']

if config['actout']=='ReLU':
    class Model_LSTM(torch.nn.Module):
        def __init__(self, input_size, d_hidden, num_layers, output_size, z_size, z_fc_size):
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
                in_features=d_hidden + z_fc_size,
                out_features=output_size
            )
            self.z_fc = torch.nn.Linear(z_size, z_fc_size)
            self.actout = torch.nn.ReLU()
            self.z_act = torch.nn.Tanh()
        
        def forward(self, x, z, hidden=None):
            if hidden is None:
                hidden = self.get_hidden(x)
            x_lstm, hidden = self.lstm(x, hidden)
            z_fc_out = self.z_act(self.z_fc(z)).unsqueeze(0).expand(x_lstm.shape[0], -1)  # shape: (seq_len, z_fc_size)
            x_concat = torch.cat([x_lstm.squeeze(0), z_fc_out], dim=1) # shape: (seq_len, d_hidden + z_fc_size)
            x_out = self.actout(self.linear(x_concat))
            return x_out, hidden
        
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
        def __init__(self, input_size, d_hidden, num_layers, output_size, z_size, z_fc_size):
            super().__init__()
            self.d_hidden = d_hidden
            self.num_layers = num_layers
            self.lstm = torch.nn.LSTM(
                input_size=input_size,
                hidden_size=d_hidden,
                num_layers=num_layers,
                batch_first=True
            )
            self.linear = torch.nn.Linear(
                in_features=d_hidden + z_fc_size,
                out_features=output_size
            )
            self.z_fc = torch.nn.Linear(z_size, z_fc_size)
            self.z_act = torch.nn.Tanh()
            self.actout = torch.nn.Softplus()
        
        def forward(self, x, z, hidden=None):
            if hidden is None:
                hidden = self.get_hidden(x)
            x_lstm, hidden = self.lstm(x, hidden)
            z_fc_out = self.z_act(self.z_fc(z)).unsqueeze(0).expand(x_lstm.shape[0], -1)  # shape: (seq_len, z_fc_size)
            x_concat = torch.cat([x_lstm.squeeze(0), z_fc_out], dim=1) # shape: (seq_len, d_hidden + z_fc_size)
            x_out = self.actout(self.linear(x_concat))
            return x_out, hidden
        
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
        def __init__(self, input_size, d_hidden, num_layers, output_size, z_size, z_fc_size):
            super().__init__()
            self.d_hidden = d_hidden
            self.num_layers = num_layers
            self.lstm = torch.nn.LSTM(
                input_size=input_size,
                hidden_size=d_hidden,
                num_layers=num_layers,
                batch_first=True
            )
            self.linear = torch.nn.Linear(
                in_features=d_hidden + z_fc_size,
                out_features=output_size
            )
            self.z_fc = torch.nn.Linear(z_size, z_fc_size)
            self.actout = torch.nn.Sigmoid()
            self.z_act = torch.nn.Tanh()
        
        def forward(self, x, z, hidden=None):
            if hidden is None:
                hidden = self.get_hidden(x)
            x_lstm, hidden = self.lstm(x, hidden)
            z_fc_out = self.z_act(self.z_fc(z)).unsqueeze(0).expand(x_lstm.shape[0], -1)  # shape: (seq_len, z_fc_size)
            # Concatenate z_fc_out to each time step in x_lstm
            x_concat = torch.cat([x_lstm.squeeze(0), z_fc_out], dim=1) # shape: (seq_len, d_hidden + z_fc_size)
            x_out = self.actout(self.linear(x_concat))
            return x_out, hidden
        
        def get_hidden(self, x):
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

# model = Model_LSTM(i_size, h_size, nb_layers, o_size) # instantiate
# model.train() # print(model)


#%% initial values
h0 = torch.zeros(nb_layers, h_size, device=device) # num_layers, hidden_size
c0 = torch.zeros(nb_layers, h_size, device=device) # num_layers, hidden_size

# torch.manual_seed(config['seed'])
tgen = torch.Generator(device=device).manual_seed(config['torch_seed'])

# # print model's state_dict
# for param_tensor in model.state_dict():
#     print(param_tensor, "\t", model.state_dict()[param_tensor].size())

state_dict_inirand = OrderedDict({
    'lstm.weight_ih_l0': torch.randn(4*h_size,i_size,device=device,generator=tgen),
    'lstm.weight_hh_l0': torch.randn(4*h_size,h_size,device=device,generator=tgen),
    'lstm.bias_ih_l0': torch.randn(4*h_size,device=device,generator=tgen),
    'lstm.bias_hh_l0': torch.randn(4*h_size,device=device,generator=tgen),
    'linear.weight': torch.randn(o_size,h_size+z_fc_size,device=device,generator=tgen),
    'linear.bias': torch.randn(o_size,device=device,generator=tgen),
    'z_fc.weight': torch.randn(z_fc_size,z_size,device=device,generator=tgen),
    'z_fc.bias': torch.randn(z_fc_size,device=device,generator=tgen)
})
# print(state_dict_inirand['linear.bias'])
# print(model.state_dict()['linear.bias'])


#%% construct Laplacian regularization matrix
len_reg = config['len_reg']
# ^ nb of pred at end of batch that are Laplacian-regularized

lapmat = np.eye(len_reg,k=-1)+np.eye(len_reg,k=1)-2*np.eye(len_reg)
lapmat[0,0] = -1.0                     # rows should add up to 0
lapmat[len_reg-1,len_reg-1] = -1.0 # rows should add up to 0
# ^ 1st order differences, to apply on y_pred for every batch

lap = torch.tensor(lapmat, dtype=torch.float32)

hp_lambda = config['lambda_LaplacianReg']/len_reg
# ^ v0.4.2: lambda scaled by len_reg so pen is mean abs diff




#%% setup for all CV folds
if config['loss']=='MSE':
    loss_fn = torch.nn.MSELoss(reduction='sum') # sum of squared errors
elif config['loss']=='MAE':
    loss_fn = torch.nn.L1Loss(reduction='sum') # sum of abs error

learning_rate = float(config['learning_rate'])
alphal2 = config['alphal2']
momentum = config['momentum']
optim = config['optim']
maxepoch = int(config['maxepoch'])

step_size = int(maxepoch/config['sch_rel_step_size'])
if step_size < 1:
    step_size = 1


ind_hor = -1 # v0.4.2: only last obs
# ^ v0.4.2: hor fixed to 1, only last obs of each batch contributes to loss
# ^ indices of obs contributing to loss eval within each batch

path_ckpt = config['path_checkpointdir'] + '/' + config['prefixoutput']
step_ckpt = config['step_ckpt'] # 10
# ^ print and record tr/va loss every maxepoch/step_ckpt

path_out_cv = config['path_outputdir'] + '/' + config['prefixoutput'] + '_cv/'
if not os.path.exists(path_out_cv):
    os.makedirs(path_out_cv) # create te output dir if does not exist


#%% setup batches within LOO-CV folds
# batches = overlaping temporal subsets over series in tr folds
# different setup from train.py: here using all time points in a series in tr,
# no tr/va splits within it since the va subset is the held-out series.

b_len = int(config['batch_len'])
b_nb = int(nT - b_len + 1) # int(nT_tr - b_len + 1)
# ^ b_nb = number of temporal batches per series, each of length b_len

# nb_batches = int((nb_series-1)*b_nb)
# ^ nb of CV tr batches, with 1 series left for each fold

# nb_param = 4*h_size*i_size + 4*h_size*h_size + 4*h_size*2 + o_size*(h_size+1)
nb_param = 4*h_size*i_size + 4*h_size*h_size + 4*h_size*2 + o_size*(h_size+z_fc_size+1) + z_fc_size*z_size + z_fc_size
nb_obs = nb_series*nT # 

# ind_tr = range(nb_batches) # index tr batches in xb/yb
# ind_va = range(b_nb) # index tr batches in xb_i/yb_i

# ind_va_i = list(range(b_len-1,nT)) # index for time points after burn-in

# nb_tr_loss = nb_batches
# nb_va_loss = b_nb

print('Number of parameters =',nb_param)
print('Total (potential) number of observations in trva =',nb_obs,'\n')
# print('Number of CV tr loss contributions =',nb_batches)
# print('Number of CV va loss contributions =',nb_va_loss,'\n')
# print('\n')



#%% CV iterations over i index
bias_tr = np.zeros(nb_series)
scale_tr = np.zeros(nb_series)
r_tr = np.zeros(nb_series)
r2_tr = np.zeros(nb_series)
MedAE_tr = np.zeros(nb_series)

bias_va = np.zeros(nb_series)
scale_va = np.zeros(nb_series)
r_va = np.zeros(nb_series)
r2_va = np.zeros(nb_series)
MedAE_va = np.zeros(nb_series)

# i = 0
for i in range(nb_series): # i index identifies held-out series
    print('--- optim CV fold',i,'/',nb_series-1,': held-out series =',seriesvec[i])
    path_best_ckpt = path_ckpt+'_ckpt_best_fold'+str(i)+'.pt'
    range_series = list(range(nb_series))
    del range_series[i] # excl i from range_series
    
    # model = Model_LSTM(i_size, h_size, nb_layers, o_size) # instantiate
    model = Model_LSTM(i_size, h_size, nb_layers, o_size, z_size, z_fc_size) # instantiate
    # model.train() # print(model)
    
    # model.load_state_dict(state_dict_inirand, strict=False)
    # # ^ <All keys matched successfully> = ok
    missing_unexpected = model.load_state_dict(state_dict_inirand, strict=False)
    if missing_unexpected.missing_keys or missing_unexpected.unexpected_keys:
        print("Warning: Some keys were missing or unexpected when loading state_dict:")
        print("  Missing keys:", missing_unexpected.missing_keys)
        print("  Unexpected keys:", missing_unexpected.unexpected_keys)
    
    # ^ <All keys matched successfully> = ok
        
    if optim=='RMSprop':
        optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr=learning_rate,
            alpha=alphal2,
            momentum=momentum
        )
    elif optim=='Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=alphal2
        )
    elif optim=='AdamW':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=alphal2
        )
    elif optim=='RAdam':
        optimizer = torch.optim.RAdam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=alphal2
        )
    
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer,
        # step_size=int(maxepoch/10), # reduce lr every 1/10 of maxepoch
        # gamma=0.9 # multiplicative factor reducing lr
        # step_size=int(maxepoch/2), # reduce lr once halfway
        # gamma=0.01 # multiplicative factor reducing lr
        # step_size=int(maxepoch/3), # shrink lr three times
        # gamma=0.1 # multiplicative factor reducing lr
        # step_size=int(maxepoch/config['sch_rel_step_size']), # 
        step_size=step_size, # 
        gamma=float(config['sch_gamma'])
    )
    
    nb_batches = int((nb_series-1)*b_nb) # reset for every fold
    xb = torch.empty(size=(nb_batches, b_len, nb_cov))
    yb = torch.empty(size=(nb_batches, b_len))
    whichseries = np.empty(nb_batches)
    for s in range(len(range_series)): # loop over series in CV tr (dim 0)
        x_s = torch.select(xfull, dim=0, index=range_series[s])
        y_s = torch.select(yfull, dim=0, index=range_series[s]) # .reshape(-1,1)
        for t in range(b_nb):
            ind_t = range(t, int(b_len+t)) # overlapping temporal subsets
            xb[t+s*b_nb,:,:] = x_s[ind_t,:]
            yb[t+s*b_nb,:] = y_s[ind_t]
            whichseries[t+s*b_nb] = range_series[s] # series in CV tr batches
    
    xb_i = torch.empty(size=(b_nb, b_len, nb_cov))
    yb_i = torch.empty(size=(b_nb, b_len))
    x_s = torch.select(xfull, dim=0, index=i)
    y_s = torch.select(yfull, dim=0, index=i) # .reshape(-1,1)
    for t in range(b_nb):
        ind_t = range(t, int(b_len+t)) # overlapping temporal subsets
        xb_i[t,:,:] = x_s[ind_t,:]
        yb_i[t,:] = y_s[ind_t]
    
    whichseries_i = np.repeat(i, b_nb) # index of series in CV va batch
    
    # xb.shape # CV tr batches
    # yb.shape # CV tr batches
    # xb_i.shape # CV held-out batches
    # yb_i.shape # CV held-out batches
    # whichseries.shape # CV tr batches
    # whichseries_i.shape # CV va batch
    
    # deal with nan in response (necessary for tstoy08)
    # ind_nonan = ~torch.any(yb.isnan(),dim=1) # bad: excl if any nan
    ind_nonan = ~yb[:,ind_hor].isnan() # good: excl if last is nan
    xb = xb[ind_nonan,:,:] # overwrite
    yb = yb[ind_nonan,:] # overwrite
    whichseries = whichseries[ind_nonan.cpu()] # overwrite
    nb_batches = xb.shape[0] # overwrite
    
    # ind_nonan = ~torch.any(yb_i.isnan(),dim=1) # bad: excl if any nan
    ind_nonan = ~yb_i[:,ind_hor].isnan() # good: excl if last is nan
    xb_i_full = xb_i # keep for full time series pred and plots
    whichseries_i_full = whichseries_i # keep for full time series pred and plots
    xb_i = xb_i[ind_nonan,:,:] # overwrite
    yb_i = yb_i[ind_nonan,:] # overwrite
    whichseries_i = whichseries_i[ind_nonan.cpu()] # overwrite
    nb_batches_i = xb_i.shape[0]
    # xb_i.shape # use to eval va loss
    # xb_i_full.shape # use for full time series pred, plots
    
    # subsample tr batches, to speed up optim
    ind_tr = np.arange(nb_batches) # indices of tr batches
    prop_tr_sub = config.get('prop_tr_sub', 1.0)  # default to 1.0 if not set
    if prop_tr_sub < 1.0:
        print('Subsampling tr batches to prop_tr_sub =',prop_tr_sub) # ,'\n'
        nb_batches_sub = int(np.floor(nb_batches*prop_tr_sub))
        if nb_batches_sub == 0:
            nb_batches_sub = 1  # at least one batch
        
        rng_trsub = np.random.default_rng(seed=int(config['srs_seed'])+1)
        # ^ different seed for tr batches subsampling, though still fixed
        ind_tr_sub = np.sort(rng_trsub.choice(ind_tr, size=nb_batches_sub, replace=False))
        
        ind_tr = ind_tr_sub # overwrite ind_tr with subsampled indices
        nb_batches = len(ind_tr) # overwrite nb_batches with subsampled number of batches
    
    
    print('Number of tr loss contributions =',nb_batches)
    print('Number of va loss contributions =',nb_batches_i)
    
    ind_tr_i = range(nb_batches) # index tr batches in xb/yb
    ind_va_i = range(nb_batches_i) # index va batches in xb_i/yb_i
    # ind_va_i = range(b_nb) # index tr batches in xb_i/yb_i
    
    
    # optim
    model.train()
    
    epoch = 0
    lossvec_tr = []
    lossvec_va = []
    epochvec = []
    
    wallclock0 = time.time()
    while (epoch < maxepoch) :
        optimizer.zero_grad()
        loss_tr = 0.0 # just to display
        loss_va = 0.0 # record va loss
        
        for b in ind_tr_i: # loop over CV tr batches
            zb_b = zb[int(whichseries[b]), :] # static cov for series b
            fwdpass = model(xb[b,:,:], zb_b, (h0,c0)) # from ini
            y_pred = fwdpass[0][-len_reg:,:] # v0.4.2
            lap_reg = torch.norm(torch.matmul(lap, y_pred),1) # sum abs diff
            y_b_tmp = yb[b,ind_hor].reshape(-1,1)
            y_pred = y_pred[ind_hor].reshape(-1,1)
            losstr = loss_fn(y_pred, y_b_tmp) + hp_lambda*lap_reg
            loss_tr += losstr.item()
            losstr.backward() # accumulate grad over batches
        
        if epoch%(maxepoch//step_ckpt)==(maxepoch//step_ckpt-1):
            with torch.no_grad():
                for b in ind_va_i: # loop over va batches (s and t)
                    zb_b = zb[int(whichseries_i[b]), :] # static cov for series b
                    fwdpass = model(xb_i[b,:,:], zb_b, (h0,c0)) # from ini
                    y_pred = fwdpass[0][ind_hor].reshape(-1,1) # horizon obs
                    loss_va += loss_fn(y_pred, yb_i[b,ind_hor].reshape(-1,1)).item()
            
            loss_tr = loss_tr/nb_batches # sum squared/absolute errors -> MSE/MAE
            loss_va = loss_va/nb_batches_i # sum squared/absolute errors -> MSE/MAE
            # save checkpoint for best intermediate fit
            if not lossvec_va: # check if empty
                torch.save(model.state_dict(), path_best_ckpt)
                epoch_best = epoch
            elif loss_va<min(lossvec_va):
                torch.save(model.state_dict(), path_best_ckpt)
                epoch_best = epoch
            
            print('epoch='+str(epoch),config['loss'],'loss: tr = {:.4f}'.format(loss_tr)+', va = {:.4f}'.format(loss_va))
            lossvec_tr.append(loss_tr)
            lossvec_va.append(loss_va)
            epochvec.append(epoch)
        
        optimizer.step() # over all series and all subsets
        scheduler.step() # update lr throughout epochs
        
        epoch += 1
        # end while
    
    wallclock1 = time.time() # in seconds
    print('while loop took',round((wallclock1 - wallclock0)/60,1),'m') # \n
    print('Smallest va loss at epoch =',epoch_best) # ,'\n'
    
    # # save estimated parameters (checkpoint) at max epoch
    # torch.save(model.state_dict(), path_ckpt + '_ckpt_' + str(epoch) + '.pt')
    
    # outputs from training and validation
    # load best set of param among epochs (best = smallest va loss)
    model.load_state_dict(torch.load(path_best_ckpt, weights_only=False))
    # ^ <All keys matched successfully> = ok
    
    model.eval()
    
    plt.figure(figsize=(12,6))
    plt.plot(epochvec, np.array(lossvec_tr), c=colvec[0], label='tr loss')
    # plt.scatter(range(maxepoch), np.array(lossvec_tr)/nT_tr, s=16,c=colvec[0])
    plt.plot(epochvec, np.array(lossvec_va), c=colvec[1], label='va loss')
    # plt.scatter(range(maxepoch), np.array(lossvec_va)/nT_va, s=16, c=colvec[1])
    plt.legend(loc='upper right')
    plt.title('tr and va '+config['loss']+' loss over all series, CV fold '+str(i))
    # plt.savefig(path_out + '_loss.pdf')
    plt.savefig(path_out_cv + 'fold' + str(i) + '_loss.pdf')
    plt.close()
    
    # assuming hor=1, so one obs per tr batch
    ytr = np.zeros(nb_batches)
    ytr_pred = np.zeros(nb_batches)
    if device.type=='cuda': # need to transfer from GPU to CPU for np
        for b in ind_tr_i: # loop over tr batches (s and t)
            zb_b = zb[int(whichseries[b]), :] # static covariates for series b
            fwdpass_b = model(xb[b,:,:], zb_b, (h0,c0)) # from ini
            ytr_pred[b] = fwdpass_b[0][ind_hor].cpu().detach().numpy().item()
            ytr[b] = yb[b,ind_hor].reshape(-1,1).cpu().detach().numpy().item()
    else: # then device.type='cpu'
        for b in ind_tr_i: # loop over tr batches (s and t)
            zb_b = zb[int(whichseries[b]), :] # static covariates for series b
            fwdpass_b = model(xb[b,:,:], zb_b, (h0,c0)) # from ini
            ytr_pred[b] = fwdpass_b[0][ind_hor].detach().numpy().item()
            ytr[b] = yb[b,ind_hor].reshape(-1,1).detach().numpy().item()
    
    # print('tr R^2 =',round(r2_score(ytr, ytr_pred),4)) # R^2 on training batches
    bias_tr[i] = np.mean(ytr) - np.mean(ytr_pred)
    scale_tr[i] = np.std(ytr)/np.std(ytr_pred)
    r_tr[i] = np.corrcoef(ytr, ytr_pred)[0,1]
    r2_tr[i] = r2_score(ytr, ytr_pred)
    MedAE_tr[i] = median_absolute_error(ytr, ytr_pred)
    
    yva = np.zeros(nb_batches_i) # b_nb
    yva_pred = np.zeros(nb_batches_i) # b_nb
    if device.type=='cuda': # need to transfer from GPU to CPU for np
        for b in ind_va_i: # loop over tr batches (s and t)
            zb_b = zb[int(whichseries_i[b]), :] # static covariates for series b
            fwdpass_b = model(xb_i[b,:,:], zb_b, (h0,c0)) # from ini
            yva_pred[b] = fwdpass_b[0][ind_hor].cpu().detach().numpy().item()
            yva[b] = yb_i[b,ind_hor].reshape(-1,1).cpu().detach().numpy().item()
    else: # then device.type='cpu'
        for b in ind_va_i: # loop over tr batches (s and t)
            zb_b = zb[int(whichseries_i[b]), :] # static covariates for series b
            fwdpass_b = model(xb_i[b,:,:], zb_b, (h0,c0)) # from ini
            yva_pred[b] = fwdpass_b[0][ind_hor].detach().numpy().item()
            yva[b] = yb_i[b,ind_hor].reshape(-1,1).detach().numpy().item()
    
    # print('va R^2 =',round(r2_score(yva, yva_pred),4)) # R^2 on validation batches
    bias_va[i] = np.mean(yva) - np.mean(yva_pred)
    scale_va[i] = np.std(yva)/np.std(yva_pred)
    r_va[i] = np.corrcoef(yva, yva_pred)[0,1]
    r2_va[i] = r2_score(yva, yva_pred)
    MedAE_va[i] = median_absolute_error(yva, yva_pred)
    
    if device.type=='cuda': # need to transfer from GPU to CPU for np
        y_s = y_s.cpu().detach().numpy()
    else: # then device.type='cpu'
        y_s = y_s.detach().numpy()
    
    # full held-out time series pred
    yva_predfull = np.zeros(b_nb) # 
    if device.type=='cuda': # need to transfer from GPU to CPU for np
        for b in range(b_nb): # loop over tr batches (s and t)
            zb_b = zb[int(whichseries_i_full[b]), :] # static cov for series b
            fwdpass_b = model(xb_i_full[b,:,:], zb_b, (h0,c0)) # from ini
            yva_predfull[b] = fwdpass_b[0][ind_hor].cpu().detach().numpy().item()
    else: # then device.type='cpu'
        for b in range(b_nb): # loop over tr batches (s and t)
            zb_b = zb[int(whichseries_i_full[b]), :] # static cov for series b
            fwdpass_b = model(xb_i_full[b,:,:], zb_b, (h0,c0)) # from ini
            yva_predfull[b] = fwdpass_b[0][ind_hor].detach().numpy().item()
    
    ind_va_i_burnin = list(range(b_len-1,nT)) # all time points after burn-in
    # np.ndarray.tolist(~np.isnan(y_s))
    # np.where(~np.isnan(y_s))
    # range(nT)[np.ndarray.tolist(~np.isnan(y_s))]
    
    plt.figure(figsize=(12,6))
    # plt.scatter(range(nT), y_s, s=10, c='grey', label='ini') # s=16
    # plt.scatter(ind_va_i_burnin, y_s[ind_va_i_burnin], s=16, c=colvec[1], label='va')
    plt.scatter(range(nT), y_s, s=16, c=colvec[1], label='va')
    plt.plot(ind_va_i_burnin, yva_predfull, linewidth=1, color='black',label='pred')
    plt.legend(loc='upper left')
    plt.title('CV fold '+str(i)+', series ' + seriesvec[i])
    plt.savefig(path_out_cv + 'fold' + str(i) + '_ts.pdf')
    plt.close()
    
    # print('\n')

# end for i in range(nb_series)

print('\n')

for i in range(nb_series): # i index identifies held-out series
    print('--- CV fold',i) # 
    print('* tr metrics:') # 
    print('  - mean(obs)-mean(pred) =',round(bias_tr[i],4)) # diff of means
    print('  - sd(obs)/sd(pred) =',round(scale_tr[i],4)) # ratio of scales
    print('  - lin corr =',round(r_tr[i],4)) # lin corr
    print('  - R^2 =',round(r2_tr[i],4)) # R^2 on te batches, by series
    print('  - MedAE =',round(MedAE_tr[i],4)) # R^2 on te batches, by series
    print('* te metrics:') # 
    print('  - mean(obs)-mean(pred) =',round(bias_va[i],4)) # diff of means
    print('  - sd(obs)/sd(pred) =',round(scale_va[i],4)) # ratio of scales
    print('  - lin corr =',round(r_va[i],4)) # lin corr
    print('  - R^2 =',round(r2_va[i],4)) # R^2 on te batches, by series
    print('  - MedAE =',round(MedAE_va[i],4)) # R^2 on te batches, by series

print('')
print('Med tr R^2 over CV folds =',round(np.median(r2_tr),4))
print('Med te R^2 over CV folds =',round(np.median(r2_va),4))
print('Ave tr R^2 over CV folds =',round(np.mean(r2_tr),4))
print('Ave te R^2 over CV folds =',round(np.mean(r2_va),4))

print('\n')
nowagain = datetime.now(tz=ZoneInfo("Europe/Zurich"))
now_again_str = nowagain.strftime("%Y-%m-%d %H:%M:%S")
print(now_again_str)
print('done')

# END twdlstm cv
