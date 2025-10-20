# twdlstm train v0.7.2

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
from sklearn.metrics import r2_score
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

# now = datetime.now() # UTC by def on runai
now = datetime.now(tz=ZoneInfo("Europe/Zurich"))
now_str = now.strftime("%Y-%m-%d %H:%M:%S")
print(now_str + ' running twdlstm train v0.7.2\n')
# print('\n')

print('Supplied config:')
print(path_config+'\n')
# print('\n')

seriesvec = config['series_trva']
covvec = config['covvec']
nT = config['nT'] # time window length, to be split in batches for tr/va

nb_series = len(seriesvec)
nb_cov = len(covvec)

# Load and stack all series (as before)
s = 0
series_s = seriesvec[s]

if config['tstoy']=='09':
    path_csv_series_s = (path_tstoy + 'SeparateSisp_tr/sisp_' + series_s + '.csv')
    dtype_dict = {
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
        'ld':float,
        'sd':float,
        'cd':float,
        'lt':float,
        'st':float
    }
else:
    path_csv_series_s = (path_tstoy + 'SeparateSeries/tstoy' + config['tstoy'] +
        '_series_' + series_s + '.csv'
    )
    dtype_dict = {
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

# 

dat_s = pd.read_csv(
    path_csv_series_s,
    header=0,
    dtype=dtype_dict
)
# print(dat_s.head())

ind_t0 = int(np.where(dat_s['ts']==config['date_t0'])[0].item())
ind_t = range(ind_t0, nT+ind_t0, 1)

y_full = dat_s['twd'][ind_t]
x_full = dat_s[covvec].iloc[ind_t,:]
x_full = np.expand_dims(x_full, axis=0)
y_full = np.expand_dims(y_full, axis=0)

for s in range(1,nb_series):
    series_s = seriesvec[s]
    if config['tstoy']=='09':
        path_csv_series_s = (path_tstoy + 'SeparateSisp_tr/sisp_' + series_s + '.csv')
    else:
        path_csv_series_s = (path_tstoy + 'SeparateSeries/tstoy' + config['tstoy'] +
            '_series_' + series_s + '.csv'
        )
    
    dat_s = pd.read_csv(
        path_csv_series_s,
        header=0,
        dtype=dtype_dict
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



#%% batches: overlaping temporal subsets over series
b_len = int(config['batch_len'])
b_nb = int(nT - b_len + 1) # int(nT_tr - b_len + 1)
# ^ b_nb = number of temporal batches per series, each of length b_len

nb_batches = int(nb_series*b_nb)

xb = torch.empty(size=(nb_batches, b_len, nb_cov))
yb = torch.empty(size=(nb_batches, b_len))
whichseries = np.empty(nb_batches)
for s in range(nb_series): # loop over series (dim 0)
    x_s = torch.select(xfull, dim=0, index=s)
    y_s = torch.select(yfull, dim=0, index=s) # .reshape(-1,1)
    for t in range(b_nb):
        ind_t = range(t, int(b_len+t)) # overlapping temporal subsets
        xb[t+s*b_nb,:,:] = x_s[ind_t,:]
        yb[t+s*b_nb,:] = y_s[ind_t]
        whichseries[t+s*b_nb] = s

# xb.shape # batches = dim 0
# yb.shape # batches = dim 0

ind_hor = -1 # pred horizon within batch = only last obs
# ^ indices of obs contributing to loss eval within each batch


#%% deal with nan in response (necessary for tstoy08-tstoy09)
ind_nonan = ~yb[:,ind_hor].isnan() # good: excl if last is nan <= same as cv.py

whichseries = whichseries[ind_nonan.cpu()] # for plotting for each series
xb = xb[ind_nonan,:,:] # overwrite
yb = yb[ind_nonan,:] # overwrite

nb_batches = xb.shape[0] # overwrite

# xb.shape # batches = dim 0
# yb.shape # batches = dim 0


#%% create tr and va subsets of batches
nb_va = int(np.floor(nb_batches*config['prop_va']))
if nb_va==0: # if no va batches, then all tr
    nb_va = 2 # at least two obs in va
    print('Warning: prop_va too small for nb obs, setting nb_va=2'+'\n')

# ^ number of batches for va, out of nb_batches
nb_tr = nb_batches - nb_va
# ^ number of batches for tr (before any subsampling, see below)
# nb_va/nb_batches # should be close to config['prop_va']

rng = np.random.default_rng(seed=int(config['srs_seed'])) # local rand generator
ind_va = np.sort(rng.choice(range(nb_batches), size=nb_va, replace=False))
ind_tr = np.array(list(set(range(nb_batches)).difference(ind_va)))
# ^ set diff: in range but not in ind_va

# print('Indices of va batches:\n',', '.join(map(str, ind_va)),'\n')
# # ^ v0.6.4: disabled printing all the va batches indices, useless


#%% subsample tr batches, to speed up optim
prop_tr_sub = config.get('prop_tr_sub', 1.0)  # default to 1.0 if not set
if prop_tr_sub < 1.0:
    print('Subsampling tr batches to prop_tr_sub =',prop_tr_sub,'\n')
    nb_tr_sub = int(np.floor(nb_tr*prop_tr_sub))
    if nb_tr_sub == 0:
        nb_tr_sub = 1  # at least one batch
    
    rng_trsub = np.random.default_rng(seed=int(config['srs_seed'])+1)
    # ^ different seed for tr batches subsampling, though still fixed
    ind_tr_sub = np.sort(rng_trsub.choice(ind_tr, size=nb_tr_sub, replace=False))
    
    ind_tr = ind_tr_sub # overwrite ind_tr with subsampled indices
    nb_tr = len(ind_tr) # overwrite nb_tr with subsampled number of batches

nb_tr_loss = nb_tr # nb_tr*hor
# ^ number of contributions to tr loss, some obs counted more than once if hor>1
#   because batches overlap in time (shifted by 1 time step from one another)
nb_va_loss = nb_va # nb_va*hor


# #%% setup static input features (z)
# zvec = config['zvec']
# z_size = len(zvec) # size of z vector = nb static input features
# z_fc_size = int(config['z_fc_size']) # size of z vector

# path_csv_static = (
#     path_tstoy + 'tstoy' + config['tstoy'] + '_staticcov.csv'
# )
# dat_static = pd.read_csv(
#     path_csv_static,
#     header=0,
#     dtype={
#         'id':str,   # id_sisp
#         'ea':float, # mch_easting
#         'no':float, # mch_northing
#         'el':float  # mch_elevation
#     }
# )

# zmat = dat_static[dat_static['id'].isin(seriesvec)] # subset series
# zmat = zmat[zvec] # keep user-supplied static features

# mean_z = np.mean(zmat, axis=0) # mean over series
# std_z = np.std(zmat, axis=0) # sd over series
# zmat = (zmat - mean_z) / std_z # normalize static features, overwrite

# # zmat.shape # nb_series, z_size

# zb = torch.tensor(zmat.values, dtype=torch.float32, device=device)
# # zb.shape # nb_series, z_size



#%% LSTM model class, initial values
i_size = nb_cov # xb.shape[2] # nb cols in x = nb input features 
d1_size = config['d1_size']
h_size = config['h_size']
d2_size = config['d2_size']
o_size = config['o_size']
nb_layers = config['nb_layers']
p_drop = config['p_drop']

h0 = torch.zeros(nb_layers, h_size, device=device) # num_layers, hidden_size
c0 = torch.zeros(nb_layers, h_size, device=device) # num_layers, hidden_size

# torch.manual_seed(config['seed'])
tgen = torch.Generator(device=device).manual_seed(int(config['torch_seed']))

if config['model']=='LSTM':
    exec(open(config['path_twdlstm'] + '/model_LSTM.py').read())
    model = Model_LSTM(i_size, h_size, nb_layers, o_size) # instantiate
    state_dict_inirand = OrderedDict({
        'lstm.weight_ih_l0': torch.randn(4*h_size, i_size, device=device,generator=tgen),
        'lstm.weight_hh_l0': torch.randn(4*h_size, h_size, device=device,generator=tgen),
        'lstm.bias_ih_l0': torch.randn(4*h_size, device=device,generator=tgen),
        'lstm.bias_hh_l0': torch.randn(4*h_size, device=device,generator=tgen),
        'linear.weight': torch.randn(o_size, h_size, device=device,generator=tgen),
        'linear.bias': torch.randn(o_size, device=device,generator=tgen)
    })
    nb_param = 4*h_size*i_size + 4*h_size*h_size + 4*h_size*2 + o_size*(h_size+1)
elif config['model']=='LSTM2':
    exec(open(config['path_twdlstm'] + '/model_LSTM2.py').read())
    model = Model_LSTM(i_size, d1_size, h_size, d2_size, nb_layers, o_size, p_drop) # instantiate
    state_dict_inirand = OrderedDict({
        'fc1.weight': torch.randn(d1_size, i_size, device=device,generator=tgen),
        'fc1.bias': torch.randn(d1_size, device=device,generator=tgen),
        'lstm.weight_ih_l0': torch.randn(4*h_size, d1_size, device=device,generator=tgen),
        'lstm.weight_hh_l0': torch.randn(4*h_size, h_size, device=device,generator=tgen),
        'lstm.bias_ih_l0': torch.randn(4*h_size, device=device,generator=tgen),
        'lstm.bias_hh_l0': torch.randn(4*h_size, device=device,generator=tgen),
        'fc2.weight': torch.randn(d2_size, h_size, device=device,generator=tgen),
        'fc2.bias': torch.randn(d2_size, device=device,generator=tgen),
        'linear.weight': torch.randn(o_size, d2_size, device=device,generator=tgen),
        'linear.bias': torch.randn(o_size, device=device,generator=tgen)
    })
    nb_param = d1_size*(i_size+1) + 4*h_size*(d1_size+h_size+2) + d2_size*(h_size+1) + o_size*(d2_size+1)

# model = Model_LSTM(i_size, h_size, nb_layers, o_size, z_size, z_fc_size) # instantiate
# model.train() # print(model)

# # print model's state_dict:
# for param_tensor in model.state_dict():
#     print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# print(state_dict_inirand['linear.bias'])
# print(model.state_dict()['linear.bias'])
missing_unexpected = model.load_state_dict(state_dict_inirand, strict=False)
if missing_unexpected.missing_keys or missing_unexpected.unexpected_keys:
    print("Warning: Some keys were missing or unexpected when loading state_dict:")
    print("  Missing keys:", missing_unexpected.missing_keys)
    print("  Unexpected keys:", missing_unexpected.unexpected_keys)

# print(missing_unexpected) # <All keys matched successfully> = ok

nb_obs = len(seriesvec)*nT # 
print('Number of parameters =',nb_param)
print('Total (potential) number of observations =',nb_obs)
print('Number of training loss contributions =',nb_tr_loss)
print('Number of validation loss contributions =',nb_va_loss,'\n')
# print('\n')


#%% setup loss and optim
if config['loss']=='MSE':
    loss_fn = torch.nn.MSELoss(reduction='sum') # sum of squared errors
elif config['loss']=='MAE':
    loss_fn = torch.nn.L1Loss(reduction='sum') # sum of abs error

learning_rate = float(config['learning_rate'])
alphal2 = float(config['alphal2'])
momentum = float(config['momentum'])
optim = config['optim']

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

maxepoch = int(config['maxepoch'])

step_size = int(maxepoch/config['sch_rel_step_size'])
if step_size < 1:
    step_size = 1

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer=optimizer,
    # step_size=int(maxepoch/10), # reduce lr every 1/10 of maxepoch
    # gamma=0.9 # multiplicative factor reducing lr
    # step_size=int(maxepoch/2), # reduce lr once halfway
    # gamma=0.01 # multiplicative factor reducing lr
    # step_size=int(maxepoch/3), # shrink lr three times
    # gamma=0.1 # multiplicative factor reducing lr
    step_size=step_size, # 
    gamma=float(config['sch_gamma'])
)


#%% construct Laplacian regularization matrix
len_reg = config['len_reg']
# ^ nb of pred at end of batch that are Laplacian-regularized

lapmat = np.eye(len_reg,k=-1)+np.eye(len_reg,k=1)-2*np.eye(len_reg)
lapmat[0,0] = -1.0                     # rows should add up to 0
lapmat[len_reg-1,len_reg-1] = -1.0 # rows should add up to 0
# lapmat # 1st order differences, to apply on y_pred for every batch

# lapmat[slice(0,int(len_reg/2)),:] = 0.0 # not pen first half of pred
# ^ v0.4.2: not necessary anymore, nb regularized pred set by len_reg 

lap = torch.tensor(lapmat, dtype=torch.float32)

hp_lambda = config['lambda_LaplacianReg']/len_reg
# ^ v0.4.2: lambda scaled by len_reg so pen is mean abs diff


#%% optim
model.train()

path_ckpt = config['path_checkpointdir'] + '/' + config['prefixoutput']
step_ckpt = config['step_ckpt'] # 10
# ^ print and record tr/va loss every maxepoch/step_ckpt

epoch = 0
lossvec_tr = []
lossvec_va = []
epochvec = []

wallclock0 = time.time()
while (epoch < maxepoch) :
    optimizer.zero_grad()
    loss_tr = 0.0 # just to display
    loss_va = 0.0 # record va loss
    # for b in range(xtr_b.shape[0]): # loop over tr batches (s and t)
    for b in ind_tr: # loop over tr batches (s and t)
        # zb_b = zb[int(whichseries[b]), :] # static covariates for series b
        # fwdpass = model(xb[b,:,:], zb_b, (h0,c0)) # added static cov
        fwdpass = model(xb[b,:,:], (h0,c0)) # from ini
        y_pred = fwdpass[0][-len_reg:,:]
        lap_reg = torch.norm(torch.matmul(lap, y_pred),1) # sum abs diff
        y_pred = y_pred[ind_hor].reshape(-1,1)
        y_b_tmp = yb[b,ind_hor].reshape(-1,1)
        losstr = loss_fn(y_pred, y_b_tmp) + hp_lambda*lap_reg
        loss_tr += losstr.item()
        losstr.backward() # accumulate grad over batches
    
    if epoch%(maxepoch//step_ckpt)==(maxepoch//step_ckpt-1):
        model.eval() # necessary with dropout layers
        with torch.no_grad():
            for b in ind_va: # loop over va batches (s and t)
                # zb_b = zb[int(whichseries[b]), :] # static covariates for series b
                # fwdpass = model(xb[b,:,:], zb_b, (h0,c0)) # added static cov
                fwdpass = model(xb[b,:,:], (h0,c0)) # from ini
                y_pred = fwdpass[0][ind_hor].reshape(-1,1) # horizon obs
                loss_va += loss_fn(y_pred, yb[b,ind_hor].reshape(-1,1)).item()
        
        loss_tr = loss_tr/nb_tr_loss # sum squared/absolute errors -> MSE/MAE
        loss_va = loss_va/nb_va_loss # sum squared/absolute errors -> MSE/MAE
        # save checkpoint for best intermediate fit
        if not lossvec_va: # check if empty
            torch.save(model.state_dict(), path_ckpt + '_ckpt_best.pt')
            epoch_best = epoch
        elif loss_va<min(lossvec_va):
            torch.save(model.state_dict(), path_ckpt + '_ckpt_best.pt')
            epoch_best = epoch
        # print('epoch='+str(epoch)+': tr',config['loss'],'loss = {:.4f}'.format(loss_tr))
        print('epoch='+str(epoch),config['loss'],'loss: tr = {:.4f}'.format(loss_tr)+', va = {:.4f}'.format(loss_va))
        lossvec_tr.append(loss_tr)
        lossvec_va.append(loss_va)
        epochvec.append(epoch)
        model.train() # back to train mode
    
    optimizer.step() # over all series and all subsets
    scheduler.step() # update lr throughout epochs
    
    epoch += 1
    # end while

wallclock1 = time.time() # in seconds
print('while loop took',round((wallclock1 - wallclock0)/60,1),'m\n')

print('Smallest va loss at epoch =',epoch_best,'\n')


# save estimated parameters (checkpoint) at max epoch
torch.save(model.state_dict(), path_ckpt + '_ckpt_' + str(epoch) + '.pt')


#%% outputs from training and validation
# load best set of param among epochs (best = smallest va loss)
model.load_state_dict(torch.load(path_ckpt+'_ckpt_best.pt', weights_only=False))
# ^ <All keys matched successfully> = ok

model.eval()

# path_out = config['path_outputdir'] + '/' + config['prefixoutput']
path_out_trva = config['path_outputdir'] + '/' + config['prefixoutput'] + '_trva/'
if not os.path.exists(path_out_trva):
    os.makedirs(path_out_trva) # create te output dir if does not exist


plt.figure(figsize=(12,6))
plt.plot(epochvec, np.array(lossvec_tr), c=colvec[0], label='tr loss')
# plt.scatter(range(maxepoch), np.array(lossvec_tr)/nT_tr, s=16,c=colvec[0])
plt.plot(epochvec, np.array(lossvec_va), c=colvec[1], label='va loss')
# plt.scatter(range(maxepoch), np.array(lossvec_va)/nT_va, s=16, c=colvec[1])
plt.legend(loc='upper right')
plt.title('training and validation '+config['loss']+' loss over all series')
# plt.savefig(path_out + '_loss.pdf')
plt.savefig(path_out_trva + 'loss.pdf')
plt.close()

# assuming hor=1, so one obs per tr batch
ytr = np.zeros(nb_tr)
ytr_pred = np.zeros(nb_tr)
if device.type=='cuda': # need to transfer from GPU to CPU for np
    for b in range(nb_tr): # loop over tr batches (s and t)
        ind_tr_b = ind_tr[b]
        # zb_b = zb[int(whichseries[ind_tr_b]), :] # static cov for series b
        # fwdpass_b = model(xb[ind_tr_b,:,:], zb_b, (h0,c0)) # from ini
        fwdpass_b = model(xb[ind_tr_b,:,:], (h0,c0)) # from ini
        ytr_pred[b] = fwdpass_b[0][ind_hor].cpu().detach().numpy().item()
        ytr[b] = yb[ind_tr_b,ind_hor].reshape(-1,1).cpu().detach().numpy().item()
else: # then device.type='cpu'
    for b in range(nb_tr): # loop over tr batches (s and t)
        ind_tr_b = ind_tr[b]
        # zb_b = zb[int(whichseries[ind_tr_b]), :] # static cov for series b
        # fwdpass_b = model(xb[ind_tr_b,:,:], zb_b, (h0,c0)) # from ini
        fwdpass_b = model(xb[ind_tr_b,:,:], (h0,c0)) # from ini
        ytr_pred[b] = fwdpass_b[0][ind_hor].detach().numpy().item()
        ytr[b] = yb[ind_tr_b,ind_hor].reshape(-1,1).detach().numpy().item()

print('tr R^2 =',round(r2_score(ytr, ytr_pred),4)) # R^2 on training batches

yva = np.zeros(nb_va)
yva_pred = np.zeros(nb_va)
if device.type=='cuda': # need to transfer from GPU to CPU for np
    for b in range(nb_va): # loop over tr batches (s and t)
        ind_va_b = ind_va[b]
        # zb_b = zb[int(whichseries[ind_va_b]), :] # static cov for series b
        # fwdpass_b = model(xb[ind_va_b,:,:], zb_b, (h0,c0)) # from ini
        fwdpass_b = model(xb[ind_va_b,:,:], (h0,c0)) # from ini
        yva_pred[b] = fwdpass_b[0][ind_hor].cpu().detach().numpy().item()
        yva[b] = yb[ind_va_b,ind_hor].reshape(-1,1).cpu().detach().numpy().item()
else: # then device.type='cpu'
    for b in range(nb_va): # loop over tr batches (s and t)
        ind_va_b = ind_va[b]
        # zb_b = zb[int(whichseries[ind_va_b]), :] # static cov for series b
        # fwdpass_b = model(xb[ind_va_b,:,:], zb_b, (h0,c0)) # from ini
        fwdpass_b = model(xb[ind_va_b,:,:], (h0,c0)) # from ini
        yva_pred[b] = fwdpass_b[0][ind_hor].detach().numpy().item()
        yva[b] = yb[ind_va_b,ind_hor].reshape(-1,1).detach().numpy().item()

print('va R^2 =',round(r2_score(yva, yva_pred),4)) # R^2 on validation batches



# count_trva = 0
# # s = 0 # 1st series as ref, plot only this one
# for s in range(len(seriesvec)): # loop over series (dim 0)
#     xfull_s = torch.select(xfull, dim=0, index=s)
#     if device.type=='cuda': # need to transfer from GPU to CPU for np
#         fwdpass_full = model(xfull_s, (h0,c0)) # from ini over all ts
#         yfull_s = torch.select(yfull, dim=0, index=s).reshape(-1,1).cpu().detach().numpy()
#         yfull_s_pred = fwdpass_full[0].cpu().detach().numpy() #
#     else: # then device.type='cpu'
#         fwdpass_full = model(xfull_s, (h0,c0)) # from ini over all ts
#         yfull_s = torch.select(yfull, dim=0, index=s).reshape(-1,1).detach().numpy()
#         yfull_s_pred = fwdpass_full[0].detach().numpy() #
    
#     ind_notnan_s = ~np.isnan(yfull_s)
#     # yfull_s = yfull_s[ind_notnan_s] # remove nan
#     # yfull_s_pred = yfull_s_pred[ind_notnan_s] # remove nan
#     print('Series',seriesvec[s],'full R^2 =',
#         round(r2_score(yfull_s[ind_notnan_s], yfull_s_pred[ind_notnan_s]),4)) # R^2 on training set
#     # yva_pred = model(xva_s, fwdpass_tr[1])[0].detach().numpy() #
#     # print('Series',seriesvec[s],'tr R^2 =',round(r2_score(ytr_s, ytr_pred),4)) # R^2 on training set
#     # print('Series',seriesvec[s],'va R^2 =',round(r2_score(yva_s, yva_pred),4)) # R^2 on validation set
    
#     # ind_tr_s = (ind_tr[whichseries[ind_tr]==s] + b_len - 1)-s*nT # np.where
#     # ind_va_s = (ind_va[whichseries[ind_va]==s] + b_len - 1)-s*nT # np.where
#     # nb_nan_burnin = sum(np.isnan(yfull_s[:b_len]))[0]
#     # ind_tr_s = ind_tr[whichseries[ind_tr]==s] + b_len - 1 - count_trva + nb_nan_burnin # s*(nT-b_len + 1)
#     # ind_va_s = ind_va[whichseries[ind_va]==s] + b_len - 1 - count_trva + nb_nan_burnin # s*(nT-b_len + 1)
#     # count_trva = count_trva + len(ind_tr_s) + len(ind_va_s)
    
#     plt.figure(figsize=(12,6))
#     plt.scatter(range(nT), yfull_s, s=10, c=colvec[0], label='obs') # s=16 # label='ini', c='grey'
#     # plt.scatter(ind_tr_s, yfull_s[ind_tr_s], s=16, c=colvec[0], label='tr')
#     # plt.scatter(ind_va_s, yfull_s[ind_va_s], s=16, c=colvec[1], label='va')
#     plt.plot(range(nT), yfull_s_pred, linewidth=1, color='black', label='fitted')
#     plt.legend(loc='upper left')
#     plt.title('series ' + seriesvec[s])
#     plt.savefig(path_out_trva + 'ts_series' + seriesvec[s] + '.pdf')
#     plt.close()


print('\n')
nowagain = datetime.now(tz=ZoneInfo("Europe/Zurich"))
now_again_str = nowagain.strftime("%Y-%m-%d %H:%M:%S")
print(now_again_str)
duration = (nowagain-now).total_seconds()
print('Time difference of',
    int(divmod(duration,86400)[0]),'day',
    int(divmod(duration,3600)[0]),'hrs',
    int(divmod(duration,60)[0]),'min',
    int(duration % 60),'sec'
)
print('done')

# source /myhome/.bashrc
# conda activate mytorch
# cd /mydata/forestcast/william/WP3
# run="00"
# nohup python -u src/twdlstm/train.py LSTM_runs/configs/config_"$run".yaml > LSTM_runs/logs/log_trva_"$run".txt &

# END twdlstm train
