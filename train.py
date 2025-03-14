# twdlstm train v0.1.1

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


#%% load tstoy data, just three series for now TODO:loop over supplied series
path_tstoy = config['path_data'] + '/tstoy' + config['tstoy'] + '/'
# '/mydata/forestcast/william/WP3/DataProcessed/tstoy04/'

# now = datetime.now() # UTC by def on runai
now = datetime.now(tz=ZoneInfo("Europe/Zurich"))
now_str = now.strftime("%Y-%m-%d %H:%M:%S")
print(now_str + ' running twdlstm train v0.1.1\n')
# print('\n')

print('Supplied config:')
print(path_config+'\n')
# print('\n')

seriesvec = config['series']

series = seriesvec[0] # '01' # 01-42
path_csv_series = (path_tstoy + 'SeparateSeries/tstoy' + config['tstoy'] +
    '_series_' + series + '.csv'
)
dat_01 = pd.read_csv(
    path_csv_series,
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

series = seriesvec[1]
path_csv_series = (path_tstoy + 'SeparateSeries/tstoy' + config['tstoy'] +
    '_series_' + series + '.csv'
)
dat_02 = pd.read_csv(
    path_csv_series,
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

series = seriesvec[2]
path_csv_series = (path_tstoy + 'SeparateSeries/tstoy' + config['tstoy'] +
    '_series_' + series + '.csv'
)
dat_03 = pd.read_csv(
    path_csv_series,
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

# print(dat_01.shape) # dimensions: rows and cols
# print(dat_02.shape) # dimensions: rows and cols
# print(dat_03.shape) # dimensions: rows and cols



#%% set tr subset with beginning of tr tw
# nT = dat_01.shape[0] # total nb time points
# print(nT)

ind_t0 = int(np.where(dat_01['ts']==config['date_t0'])[0].item())
# ^ dat row index corresponding to date_t0 in config
# print(dat_01.iloc[ind_t0,:]) # check

nT_tr = config['nT_tr'] # size of tr set
nT_va = config['nT_va'] # size of va set

# maxlag = 0 #
# shift = 0
# n_cont = 0

ind_tr = range(ind_t0, nT_tr+ind_t0, 1) # for i in ind_tr: print(i)
ind_va = range(nT_tr+ind_t0, nT_tr+ind_t0+nT_va, 1) # for i in ind_va: print(i)

# ind_shift_tr = range(ind_t0-shift-n_cont, ind_t0-shift+nT_tr, 1)
# ind_shift_va = range(ind_t0-shift-n_cont+nT_tr, ind_t0-shift+nT_tr+nT_va, 1)
# # ind_tr is the ref, shift and c_cont only for cov

y_01_full = dat_01['twd']
y_01_tr = y_01_full[ind_tr]
y_01_va = y_01_full[ind_va]

y_02_full = dat_02['twd']
y_02_tr = y_02_full[ind_tr]
y_02_va = y_02_full[ind_va]

y_03_full = dat_03['twd']
y_03_tr = y_03_full[ind_tr]
y_03_va = y_03_full[ind_va]

# print(y_01_tr.isna().any())
# print(y_01_va.isna().any())
# print(y_02_tr.isna().any())
# print(y_02_va.isna().any())
# print(y_03_tr.isna().any())
# print(y_03_va.isna().any())
# # ^ all have no NAs, ok

covvec = config['covvec']

x_01_full = dat_01[covvec]
x_02_full = dat_02[covvec]
x_03_full = dat_03[covvec]
x_01_tr = x_01_full.iloc[ind_tr,:] # ini, time subset, all cols
x_01_va = x_01_full.iloc[ind_va,:] # ini, time subset, all cols
x_02_tr = x_02_full.iloc[ind_tr,:] # ini, time subset, all cols
x_02_va = x_02_full.iloc[ind_va,:] # ini, time subset, all cols
x_03_tr = x_03_full.iloc[ind_tr,:] # ini, time subset, all cols
x_03_va = x_03_full.iloc[ind_va,:] # ini, time subset, all cols

# print(x_01_tr.isna().any())
# print(x_02_tr.isna().any())
# print(x_03_tr.isna().any())
# # ^ ok, no NAs

# x_normalized = False
# y_01_tr.shape
# x_01_tr.shape
# y_01_va.shape
# x_01_va.shape
# # ^ nb cols for x = len(covvec)*(maxlag+1)

# plt.figure(figsize=(12,6))
# plt.scatter(ind_tr, y_01_tr, s=16, c=colvec[0], label='tr')
# plt.scatter(ind_va, y_01_va, s=16, c=colvec[1], label='va')
# plt.legend(loc='upper left')
# plt.title('series 01')
# plt.savefig('plot.pdf')
# plt.close()





# #%% normalize input features for mean=0 and var=1
mean_01 = np.apply_along_axis(np.mean, 0, x_01_tr)
mean_02 = np.apply_along_axis(np.mean, 0, x_02_tr)
mean_03 = np.apply_along_axis(np.mean, 0, x_03_tr)
sd_01 = np.apply_along_axis(np.std, 0, x_01_tr)
sd_02 = np.apply_along_axis(np.std, 0, x_02_tr)
sd_03 = np.apply_along_axis(np.std, 0, x_03_tr)

for j in range(x_01_tr.shape[1]): # loop over columns, overwrite each cov
    x_01_tr.iloc[:,j] = (x_01_tr.iloc[:,j]-mean_01[j])/sd_01[j]
    x_01_va.iloc[:,j] = (x_01_va.iloc[:,j]-mean_01[j])/sd_01[j]
    x_02_tr.iloc[:,j] = (x_02_tr.iloc[:,j]-mean_02[j])/sd_02[j]
    x_02_va.iloc[:,j] = (x_02_va.iloc[:,j]-mean_02[j])/sd_02[j]
    x_03_tr.iloc[:,j] = (x_03_tr.iloc[:,j]-mean_03[j])/sd_03[j]
    x_03_va.iloc[:,j] = (x_03_va.iloc[:,j]-mean_03[j])/sd_03[j]

x_normalized = True

x01tr = torch.tensor(x_01_tr.values, dtype=torch.float32)
x02tr = torch.tensor(x_02_tr.values, dtype=torch.float32)
x03tr = torch.tensor(x_03_tr.values, dtype=torch.float32)
x01va = torch.tensor(x_01_va.values, dtype=torch.float32)
x02va = torch.tensor(x_02_va.values, dtype=torch.float32)
x03va = torch.tensor(x_03_va.values, dtype=torch.float32)

y01tr = torch.tensor(y_01_tr.values, dtype=torch.float32).reshape(-1,1)
y02tr = torch.tensor(y_02_tr.values, dtype=torch.float32).reshape(-1,1)
y03tr = torch.tensor(y_03_tr.values, dtype=torch.float32).reshape(-1,1)
y01va = torch.tensor(y_01_va.values, dtype=torch.float32).reshape(-1,1)
y02va = torch.tensor(y_02_va.values, dtype=torch.float32).reshape(-1,1)
y03va = torch.tensor(y_03_va.values, dtype=torch.float32).reshape(-1,1)

# y01tr.shape
# x01tr.shape
# y01va.shape
# x01va.shape



#%% LSTM  model class
i_size = x01tr.shape[1] # nb cols in xtr = nb input features 
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
model.train() # print(model)


#%% initial values
torch.manual_seed(config['seed'])
# torch.manual_seed(123)
state_dict_inirand = OrderedDict({
    'lstm.weight_ih_l0': torch.randn(4*h_size,i_size),
    'lstm.weight_hh_l0': torch.randn(4*h_size,h_size),
    'lstm.bias_ih_l0': torch.randn(4*h_size),
    'lstm.bias_hh_l0': torch.randn(4*h_size),
    'linear.weight': torch.randn(o_size,h_size),
    'linear.bias': torch.randn(o_size)
})
# print(state_dict_inirand['linear.bias']) # -0.6977 if seed=123
model.load_state_dict(state_dict_inirand, strict=False)
# ^ <All keys matched successfully> = ok

# h0 = torch.randn(nb_layers, h_size) # num_layers, hidden_size
# c0 = torch.randn(nb_layers, h_size) # num_layers, hidden_size
h0 = torch.zeros(nb_layers, h_size) # num_layers, hidden_size
c0 = torch.zeros(nb_layers, h_size) # num_layers, hidden_size

nb_param = 4*h_size*i_size + 4*h_size*h_size + 4*h_size*2 + o_size*(h_size+1)
nb_obs = 3*nT_tr # 3 because 3 series on config yaml
print('Total number of parameters =',nb_param)
print('Total number of training observations =',nb_obs,'\n')
# print('\n')


#%% setup loss and optim
loss_fn = torch.nn.MSELoss(reduction='sum') # TODO: add loss in config
# loss_fn = torch.nn.MSELoss(reduction='mean')
# loss_fn = torch.nn.L1Loss(reduction='mean') # mean asbolute error
# loss_fn = torch.nn.L1Loss(reduction='sum') # sum of abs error

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


#%% optim
epoch = 0
lossvec_tr = []
lossvec_va = []

wallclock0 = time.time()
while (epoch < maxepoch) :
    # model.train()
    optimizer.zero_grad()
    loss_tr = 0.0 # just to display
    loss_va = 0.0 # record va loss
    # for j in range(3): # loop over all tr obs
    # 01
    fwdpass = model(x01tr,(h0,c0)) # from ini
    y_pred = fwdpass[0] # whole series at once
    losstr = loss_fn(y_pred, y01tr)
    loss_tr += losstr.item()
    losstr.backward() # accumulate grad over series
    with torch.no_grad():
        yva_pred = model(x01va, fwdpass[1])[0]
        loss_va += loss_fn(yva_pred, y01va).item()
    # 02
    fwdpass = model(x02tr,(h0,c0)) # from ini
    y_pred = fwdpass[0] # whole series at once
    losstr = loss_fn(y_pred, y02tr)
    loss_tr += losstr.item()
    losstr.backward() # accumulate grad over series
    with torch.no_grad():
        yva_pred = model(x01va, fwdpass[1])[0]
        loss_va += loss_fn(yva_pred, y01va).item()
    # 03
    fwdpass = model(x03tr,(h0,c0)) # from ini
    y_pred = fwdpass[0] # whole series at once
    losstr = loss_fn(y_pred, y03tr)
    loss_tr += losstr.item()
    losstr.backward() # accumulate grad over series
    with torch.no_grad():
        yva_pred = model(x01va, fwdpass[1])[0]
        loss_va += loss_fn(yva_pred, y01va).item()
    
    optimizer.step() # over all series
    # scheduler.step() # update lr throughout epochs
    
    # if epoch%(maxepoch/10)==(maxepoch/10-1):
    print('epoch='+str(epoch)+': tr loss = {:.4f}'.format(loss_tr))
    lossvec_tr.append(loss_tr)
    lossvec_va.append(loss_va)
    # if epoch==46: # smallest va loss with h_size=32
    #     break
    epoch += 1
    # end while

wallclock1 = time.time() # in seconds
print('while loop took',round((wallclock1 - wallclock0)/60,1),'m\n')
# print('\n')

# grads = []
# for param in model.parameters():
#     grads.append(param.grad.view(-1))

# grads = torch.cat(grads) # print(grads.shape) # = nb_param
# print('max abs grad =',round(max(np.abs(grads)).item(),2))


#%% outputs from training and validation
path_out = config['path_outputdir'] + '/' + config['prefixoutput']

plt.figure(figsize=(12,6))
plt.plot(range(maxepoch), np.array(lossvec_tr)/nT_tr, c=colvec[0], label='tr loss')
# plt.scatter(range(maxepoch), np.array(lossvec_tr)/nT_tr, s=16,c=colvec[0])
plt.plot(range(maxepoch), np.array(lossvec_va)/nT_va, c=colvec[1], label='va loss')
# plt.scatter(range(maxepoch), np.array(lossvec_va)/nT_va, s=16, c=colvec[1])
plt.legend(loc='upper right')
plt.title('training and validation loss over all series')
plt.savefig(path_out + '_loss.pdf')
plt.close()

model.eval()

fwdpass_tr = model(x01tr,(h0,c0)) # from ini over all ts
ytr_pred = fwdpass_tr[0].detach().numpy() #
yva_pred = model(x01va, fwdpass_tr[1])[0].detach().numpy() #
print('Series',seriesvec[0],'tr R^2 =',round(r2_score(y_01_tr, ytr_pred),4)) # R^2 on training set
print('Series',seriesvec[0],'va R^2 =',round(r2_score(y_01_va, yva_pred),4)) # R^2 on validation set

plt.figure(figsize=(12,6))
plt.scatter(ind_tr, y_01_tr, s=16, c=colvec[0], label='tr')
plt.scatter(ind_va, y_01_va, s=16, c=colvec[1], label='va')
plt.plot(ind_tr, ytr_pred, linewidth=1, color=colvec[0])
plt.plot(ind_va, yva_pred, linewidth=1, color=colvec[1])
plt.legend(loc='upper left')
plt.title('series ' + seriesvec[0])
plt.savefig(path_out + '_pred_series' + seriesvec[0] + '.pdf')
plt.close()

print('\n')
print('done')

# END twdlstm train
