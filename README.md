twdlstm: PyTorch code for training an LSTM NN on TWD_norm series
----------------------------------------------------------------

### TODO


### Version history

This is twdlstm version 0.6.5. Change log:
* v0.6.5: cv.py, commented-out everything related to static input features (z), need to add option to disable it in future versions to check, e.g., elevation, helps CV pred.
* v0.6.4:
  - train.py, cv.py, pred.py: added static input features (z), passed to a fully connected layer, a tanh activation function, and then concatenated to the LSTM hidden state before the last linear layer and output activation.
* v0.6.3:
  - train.py, cv.py, pred.py: cov mean/sd normalization now done for all series/time points/grid points taken together.
  - train.py, cv.py: added tr batches subsampling with config argument prop_tr_sub. If prop_tr_sub=1.0 then no subsampling (using all tr available obs, like before).
* v0.6.2:
  - pred.py adapted for additional day of the year (dy) covariate (otherwise worked as is since v0.5.3). test.py and config.yaml remain at v0.5.3 for now.
  - in train.py and cv.py, lr scheduling now in config options (sch_rel_step_size and sch_gamma). Scheduling effectively disabled if sch_rel_step_size=1 or sch_gamma=1.
* v0.6.1: in train.py and cv.py, added learning rate scheduler. Hard-coded to shrink lr three times by a factor of 0.1. So setting learning_rate to 1e-1 in config is prefereable.
* v0.6:
  - in train.py, simplified the series plots, only showing the full time series with fitted values computed in a single forward pass.
  - train.py and cv.py adapted to work with incompete time series (arbitrary nan like in tstoy08). pred.py and test.py left as of v0.5.3 for now.
* v0.5.3: in pred.py, fixed covariate normalization (using mean and sd over grid from same day) and added 'vp' and 'sw' covariates names.
* v0.5.2:
  - in pred.py, new config argument source allows to select between 'train' (output from train.py, single fit to all tr data) and 'cv' (output from cv.py, one fit per CV fold).
  - in pred.py, new config argument whichckpt allows to select between 'best' (smallest va loss thought epochs) and 'last' (last epoch in optim). 'last' only available for source='train'. 'best' is the default. 
* v0.5.1:
  - created pred.py script. It predicts the TWD response on Jan's CH grid, outputs a zarr file with: dim 1 = LOO-CV folds, one fold being an "ensemble member", dim 2 = grid x coordinates, and dim 3 = grid y coordinates.
  - all .py scripts: fixed printed time at end, now correct time difference
  - all .py scrpts: added option for different output activation function, with "ReLU", "Softplus", and "Sigmoid" currently implemented with argument actout in config.yaml.
* v0.5:
  - fixed dtype in pd.read_csv for new covariates sw, dy, and el.
  - created cv.py. Uses same config as train.py and test.py, runs train on all sites in series_trva except one, one at a time, i.e. a LOO CV. Computes some metrics by site (same as the ones in test.py) but does not save all CV preds for now.
* v0.4.4:
  - removed MAPE from test.py metrics, useless
  - create sub-dir for trva output, plot ts full pred for all trva series now
* v0.4.3:
  - added optim choice in config yaml, for train.py.
  - in train.py, now loading best param over epoch (and not keeping the last one) for outputs
* v0.4.2: changed scheme for Laplacian regularization in train.py. Now hor is hard-coded to 1 (last obs in batch contributes to loss) and len_reg in config is the number of pred at the end of every batch that are Laplacian-regularized. Also, hp_lambda is now scaled by len_reg so the penalty is the mean of absolute differences.
* v0.4.1: added Laplacian regularizer (sum of abs diff of pred within batch) in train.py, with lambda_LaplacianReg hyperparameter supplied in config.
* v0.4:
  - adapted train.py for series_trva (distinct now from series_te in config)
  - created test.py to deploy fitted LSTM (from train.py) on te series
* v0.3.2:
  - added path_checkpointdir and step_ckpt to config
  - print and record tr/va loss every maxepoch/step_ckpt
  - save model param at maxepoch as checkpoint (regardless of performance)
  - save best param (smallest va loss) as checkpoint
* v0.3.1: adapted train.py to run on CPU or GPU. No user option, if GPU available then cuda by default for all torch tensors (including randn).
* v0.3: completely changed tr and va subsets. Entire time window (set by nT length in config) is split in batches of size batch_len (supplied in config), tr and va batches are randomly selected over all series with va size being specified by prop_va in config, and tr and va losses are only being evaluated at last loss_hor observations (also supplied in config, should stay 1 though otherwise overlap) within each batch.
* v0.2.1:
  - fixed total number of parameters printed in log
  - added loss to config, choice betwen 'MSE' and 'MAE'
* v0.2: arbitrary number of series can be supplied (>=2), loop over seriesvec
* v0.1.1: fixed datetime stamp for Europe/Zurich tz
* v0.1: initial release
