# twdlstm train v0.4.1

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
print(now_str + ' running twdlstm train v0.4.1\n')
# print('\n')

print('Supplied config:')
print(path_config+'\n')
# print('\n')

# seriesvec = config['series'] # up to v0.3.2
seriesvec = config['series_trva'] # as of v0.4, distinguish from series_te
covvec = config['covvec']
# nT_tr = config['nT_tr'] # size of tr set
# nT_va = config['nT_va'] # size of va set
nT = config['nT'] # time window length, to be split in batches for tr/va

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
        'lr':float
    }
)

ind_t0 = int(np.where(dat_s['ts']==config['date_t0'])[0].item())
# ^ dat row index corresponding to date_t0 in config

# ind_tr = range(ind_t0, nT_tr+ind_t0, 1) # for i in ind_tr: print(i)
# ind_va = range(nT_tr+ind_t0, nT_tr+ind_t0+nT_va, 1) # for i in ind_va: print(i)
ind_t = range(ind_t0, nT+ind_t0, 1) # for i in ind_t: print(i)

# y_full = dat_s['twd']
# y_tr = y_full[ind_tr]
# y_va = y_full[ind_va]
# x_full = dat_s[covvec]
# x_tr = x_full.iloc[ind_tr,:] # ini, time subset, all cols
# x_va = x_full.iloc[ind_va,:] # ini, time subset, all cols

y_full = dat_s['twd'][ind_t]
x_full = dat_s[covvec].iloc[ind_t,:] # ini, time subset, all cols

# mean_s = np.apply_along_axis(np.mean, 0, x_tr) # tr cov mean
# sd_s = np.apply_along_axis(np.std, 0, x_tr) # tr cov sd
# for j in range(x_tr.shape[1]): # loop over columns, overwrite each cov
#     x_tr.iloc[:,j] = (x_tr.iloc[:,j]-mean_s[j])/sd_s[j]
#     x_va.iloc[:,j] = (x_va.iloc[:,j]-mean_s[j])/sd_s[j]

# v0.3: use entire time window (tr and va batches) for cov norm
mean_s = np.apply_along_axis(np.mean, 0, x_full) # tr cov mean
sd_s = np.apply_along_axis(np.std, 0, x_full) # tr cov sd
for j in range(nb_cov): # loop over columns, overwrite each cov
    x_full.iloc[:,j] = (x_full.iloc[:,j]-mean_s[j])/sd_s[j]


# x_tr = np.expand_dims(x_tr, axis=0) # add a dim for stacking series
# x_va = np.expand_dims(x_va, axis=0) # add a dim for stacking series
# y_tr = np.expand_dims(y_tr, axis=0) # add a dim for stacking series
# y_va = np.expand_dims(y_va, axis=0) # add a dim for stacking series
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
            'lr':float
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

# x_normalized = True

# xtr = torch.tensor(x_tr, dtype=torch.float32)
# xva = torch.tensor(x_va, dtype=torch.float32)
# ytr = torch.tensor(y_tr, dtype=torch.float32) # .reshape(-1,1)
# yva = torch.tensor(y_va, dtype=torch.float32)
# xtr.shape
# xva.shape
# ytr.shape
# yva.shape


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
for s in range(nb_series): # loop over series (dim 0)
    x_s = torch.select(xfull, dim=0, index=s)
    y_s = torch.select(yfull, dim=0, index=s) # .reshape(-1,1)
    for t in range(b_nb):
        ind_t = range(t, int(b_len+t)) # overlapping temporal subsets
        xb[t+s*b_nb,:,:] = x_s[ind_t,:]
        yb[t+s*b_nb,:] = y_s[ind_t]

# xb.shape # batches = dim 0
# yb.shape # batches = dim 0

hor = config['loss_hor']
ind_hor = range(int(b_len-hor),b_len)
# ^ indices of obs contributing to loss eval within each batch


#%% create tr and va subsets of batches
nb_va = int(np.floor(nb_batches*config['prop_va']))
# ^ number of batches for va, out of nb_batches
nb_tr = nb_batches - nb_va
# ^ number of batches for tr
# nb_va/nb_batches # should be close to config['prop_va']

# np.random.seed(seed=int(config['srs_seed'])) # fix seed simple random sampling
# np.random.choice(range(5), size=2, replace=False) # global seed fixed
rng = np.random.default_rng(seed=int(config['srs_seed'])) # local rand generator
ind_va = np.sort(rng.choice(range(nb_batches), size=nb_va, replace=False))
ind_tr = np.array(list(set(range(nb_batches)).difference(ind_va)))
# ^ set diff: in range but not in ind_va

print('Indices of va batches:\n',', '.join(map(str, ind_va)),'\n')

nb_tr_loss = nb_tr*hor
# ^ number of contributions to tr loss, some obs counted more than once if hor>1
#   because batches overlap in time (shifted by 1 time step from one another)
nb_va_loss = nb_va*hor



#%% LSTM model class
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
# path_ckpt = '/mydata/forestcast/william/WP3/LSTM_runs/checkpoints/00_ckpt_10.pt'
# model.load_state_dict(torch.load(path_ckpt, weights_only=False)) # checkpoint

h0 = torch.zeros(nb_layers, h_size) # num_layers, hidden_size
c0 = torch.zeros(nb_layers, h_size) # num_layers, hidden_size

# torch.manual_seed(config['seed'])
tgen = torch.Generator(device=device).manual_seed(config['torch_seed'])

# state_dict_inirand = OrderedDict({
#     'lstm.weight_ih_l0': torch.randn(4*h_size,i_size),
#     'lstm.weight_hh_l0': torch.randn(4*h_size,h_size),
#     'lstm.bias_ih_l0': torch.randn(4*h_size),
#     'lstm.bias_hh_l0': torch.randn(4*h_size),
#     'linear.weight': torch.randn(o_size,h_size),
#     'linear.bias': torch.randn(o_size)
# })

state_dict_inirand = OrderedDict({
    'lstm.weight_ih_l0': torch.randn(4*h_size,i_size,device=device,generator=tgen),
    'lstm.weight_hh_l0': torch.randn(4*h_size,h_size,device=device,generator=tgen),
    'lstm.bias_ih_l0': torch.randn(4*h_size,device=device,generator=tgen),
    'lstm.bias_hh_l0': torch.randn(4*h_size,device=device,generator=tgen),
    'linear.weight': torch.randn(o_size,h_size,device=device,generator=tgen),
    'linear.bias': torch.randn(o_size,device=device,generator=tgen)
})
# print(state_dict_inirand['linear.bias'])
# print(model.state_dict()['linear.bias'])
model.load_state_dict(state_dict_inirand, strict=False)
# ^ <All keys matched successfully> = ok

nb_param = 4*h_size*i_size + 4*h_size*h_size + 4*h_size*2 + o_size*(h_size+1)
nb_obs = len(seriesvec)*nT # 
print('Number of parameters =',nb_param)
print('Total number of observations =',nb_obs)
print('Number of training loss contributions =',nb_tr_loss)
print('Number of validation loss contributions =',nb_va_loss,'\n')
# print('\n')


#%% setup loss and optim
if config['loss']=='MSE':
    loss_fn = torch.nn.MSELoss(reduction='sum') # sum of squared errors
elif config['loss']=='MAE':
    loss_fn = torch.nn.L1Loss(reduction='sum') # sum of abs error

# loss_fn = torch.nn.MSELoss(reduction='mean') # mean squared error
# loss_fn = torch.nn.L1Loss(reduction='mean') # mean asbolute error


learning_rate = float(config['learning_rate'])
alphal2 = config['alphal2']
momentum = config['momentum']

optimizer = torch.optim.RMSprop(
    model.parameters(),
    lr=learning_rate,
    alpha=alphal2,
    momentum=momentum
)
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# TODO: add optim in config

maxepoch = int(config['maxepoch'])

# scheduler = torch.optim.lr_scheduler.StepLR(
#     optimizer=optimizer,
#     # step_size=int(maxepoch/10), # reduce lr every 1/10 of maxepoch
#     # gamma=0.9 # multiplicative factor reducing lr
#     step_size=int(maxepoch/2), # reduce lr once halfway
#     gamma=0.1 # multiplicative factor reducing lr
#     # step_size=int(maxepoch/4), #
#     # gamma=0.5 # multiplicative factor reducing lr
#     # step_size=int(maxepoch/4), # 
#     # gamma=0.1 # multiplicative factor reducing lr
# )
# TODO: add scheduler


#%% construct Laplacian regularization matrix
step_ckpt = config['step_ckpt'] # 10

lapmat = np.eye(step_ckpt,k=-1)+np.eye(step_ckpt,k=1)-2*np.eye(step_ckpt)
lapmat[0,0] = -1.0                     # rows should add up to 0
lapmat[step_ckpt-1,step_ckpt-1] = -1.0 # rows should add up to 0
# lapmat # 1st order differences, to apply on y_pred for every batch

lapmat[slice(0,int(step_ckpt/2)),:] = 0.0 # not pen first half of pred
# lapmat

lap = torch.tensor(lapmat, dtype=torch.float32)

hp_lambda = config['lambda_LaplacianReg']


#%% optim
model.train()

path_ckpt = config['path_checkpointdir'] + '/' + config['prefixoutput']
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
        fwdpass = model(xb[b,:,:], (h0,c0)) # from ini
        y_pred = fwdpass[0] # eval loss only on horizon obs
        lap_reg = torch.norm(torch.matmul(lap, y_pred),1) # sum abs diff
        y_b_tmp = yb[b,ind_hor].reshape(-1,1)
        losstr = loss_fn(y_pred[ind_hor], y_b_tmp) + hp_lambda*lap_reg
        loss_tr += losstr.item()
        losstr.backward() # accumulate grad over batches
    
    optimizer.step() # over all series and all subsets
    # scheduler.step() # update lr throughout epochs
    
    if epoch%(maxepoch/step_ckpt)==(maxepoch/step_ckpt-1):
        with torch.no_grad():
            for b in ind_va: # loop over va batches (s and t)
                fwdpass = model(xb[b,:,:], (h0,c0)) # from ini
                y_pred = fwdpass[0][ind_hor] # eval loss only on horizon obs
                loss_va += loss_fn(y_pred, yb[b,ind_hor].reshape(-1,1)).item()
        # loss_tr = loss_tr/nb_obs # sum squared/absolute errors -> MSE/MAE
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
    
    epoch += 1
    # end while

wallclock1 = time.time() # in seconds
print('while loop took',round((wallclock1 - wallclock0)/60,1),'m\n')

print('Smallest va loss at epoch =',epoch_best,'\n')


# save estimated parameters (checkpoint)
torch.save(model.state_dict(), path_ckpt + '_ckpt_' + str(epoch) + '.pt')


#%% outputs from training and validation
model.eval()

path_out = config['path_outputdir'] + '/' + config['prefixoutput']

plt.figure(figsize=(12,6))
plt.plot(epochvec, np.array(lossvec_tr), c=colvec[0], label='tr loss')
# plt.scatter(range(maxepoch), np.array(lossvec_tr)/nT_tr, s=16,c=colvec[0])
plt.plot(epochvec, np.array(lossvec_va), c=colvec[1], label='va loss')
# plt.scatter(range(maxepoch), np.array(lossvec_va)/nT_va, s=16, c=colvec[1])
plt.legend(loc='upper right')
plt.title('training and validation '+config['loss']+' loss over all series')
plt.savefig(path_out + '_loss.pdf')
plt.close()

# assuming hor=1, so one obs per tr batch
ytr = np.zeros(nb_tr)
ytr_pred = np.zeros(nb_tr)
if device.type=='cuda': # need to transfer from GPU to CPU for np
    for b in range(nb_tr): # loop over tr batches (s and t)
        ind_tr_b = ind_tr[b]
        fwdpass_b = model(xb[ind_tr_b,:,:], (h0,c0)) # from ini
        ytr_pred[b] = fwdpass_b[0][ind_hor].cpu().detach().numpy().item()
        ytr[b] = yb[ind_tr_b,ind_hor].reshape(-1,1).cpu().detach().numpy().item()
else: # then device.type='cpu'
    for b in range(nb_tr): # loop over tr batches (s and t)
        ind_tr_b = ind_tr[b]
        fwdpass_b = model(xb[ind_tr_b,:,:], (h0,c0)) # from ini
        ytr_pred[b] = fwdpass_b[0][ind_hor].detach().numpy().item()
        ytr[b] = yb[ind_tr_b,ind_hor].reshape(-1,1).detach().numpy().item()

print('tr R^2 =',round(r2_score(ytr, ytr_pred),4)) # R^2 on training batches

yva = np.zeros(nb_va)
yva_pred = np.zeros(nb_va)
if device.type=='cuda': # need to transfer from GPU to CPU for np
    for b in range(nb_va): # loop over tr batches (s and t)
        ind_va_b = ind_va[b]
        fwdpass_b = model(xb[ind_va_b,:,:], (h0,c0)) # from ini
        yva_pred[b] = fwdpass_b[0][ind_hor].cpu().detach().numpy().item()
        yva[b] = yb[ind_va_b,ind_hor].reshape(-1,1).cpu().detach().numpy().item()
else: # then device.type='cpu'
    for b in range(nb_va): # loop over tr batches (s and t)
        ind_va_b = ind_va[b]
        fwdpass_b = model(xb[ind_va_b,:,:], (h0,c0)) # from ini
        yva_pred[b] = fwdpass_b[0][ind_hor].detach().numpy().item()
        yva[b] = yb[ind_va_b,ind_hor].reshape(-1,1).detach().numpy().item()

print('va R^2 =',round(r2_score(yva, yva_pred),4)) # R^2 on validation batches

s = 0 # 1st series as ref, plot only this one
xfull_s = torch.select(xfull, dim=0, index=s)
if device.type=='cuda': # need to transfer from GPU to CPU for np
    fwdpass_full = model(xfull_s, (h0,c0)) # from ini over all ts
    yfull_s = torch.select(yfull, dim=0, index=s).reshape(-1,1).cpu().detach().numpy()
    yfull_s_pred = fwdpass_full[0].cpu().detach().numpy() #
else: # then device.type='cpu'
    fwdpass_full = model(xfull_s, (h0,c0)) # from ini over all ts
    yfull_s = torch.select(yfull, dim=0, index=s).reshape(-1,1).detach().numpy()
    yfull_s_pred = fwdpass_full[0].detach().numpy() #

print('Series',seriesvec[s],'full R^2 =',round(r2_score(yfull_s, yfull_s_pred),4)) # R^2 on training set
# yva_pred = model(xva_s, fwdpass_tr[1])[0].detach().numpy() #
# print('Series',seriesvec[s],'tr R^2 =',round(r2_score(ytr_s, ytr_pred),4)) # R^2 on training set
# print('Series',seriesvec[s],'va R^2 =',round(r2_score(yva_s, yva_pred),4)) # R^2 on validation set

ind_01_tr = [0] # ini
for b in range(nb_tr): # loop over tr batches (s and t)
    ind_tr_b = ind_tr[b]
    if ind_tr_b < b_nb: # then batch in 1st series
        ind_01_tr.append(ind_tr_b + b_len - 1) # index of obs contributing to loss

ind_01_tr = ind_01_tr[1:] # excl ini 0

ind_01_va = [0] # ini
for b in range(nb_va): # loop over tr batches (s and t)
    ind_va_b = ind_va[b]
    if ind_va_b < b_nb: # then batch in 1st series
        ind_01_va.append(ind_va_b + b_len -1 ) # index of obs contributing to loss

ind_01_va = ind_01_va[1:] # excl ini 0

plt.figure(figsize=(12,6))
plt.scatter(range(nT), yfull_s, s=10, c='grey', label='ini') # s=16
plt.scatter(ind_01_tr, yfull_s[ind_01_tr], s=16, c=colvec[0], label='tr')
plt.scatter(ind_01_va, yfull_s[ind_01_va], s=16, c=colvec[1], label='va')
plt.plot(range(nT), yfull_s_pred, linewidth=1, color='black')
plt.legend(loc='upper left')
plt.title('series ' + seriesvec[s])
plt.savefig(path_out + '_pred_series' + seriesvec[s] + '.pdf')
plt.close()

for s in range(1,len(seriesvec)): # loop over series (dim 0)
    xfull_s = torch.select(xfull, dim=0, index=s)
    fwdpass_full = model(xfull_s, (h0,c0)) # from ini over all ts
    if device.type=='cuda': # need to transfer from GPU to CPU for np
        yfull_s = torch.select(yfull,dim=0,index=s).reshape(-1,1).cpu().detach().numpy()
        yfull_s_pred = fwdpass_full[0].cpu().detach().numpy() #
    else: # then device.type='cpu'
        yfull_s = torch.select(yfull, dim=0, index=s).reshape(-1,1).detach().numpy()
        yfull_s_pred = fwdpass_full[0].detach().numpy() #
    print('Series',seriesvec[s],'full R^2 =',round(r2_score(yfull_s, yfull_s_pred),4)) # R^2 on training set



print('\n')
print('done')

# END twdlstm train
