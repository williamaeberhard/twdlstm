twdlstm: PyTorch code for training an LSTM NN on TWD_norm series
----------------------------------------------------------------

### TODO


### Version history

This is twdlstm version 0.4.3. Change log:
* v0.4.3:
  - added optim choice in config yaml, for train.py.
  - in train.py, now loading best param over epoch (and not keeping the last one) for outputs
* v0.4.2: changed scheme for Laplacian regularization in train.py. Now hor is hard-coded to 1 (last obs in batch contributes to loss) and len_reg in config is the number of pred at the end of every batch that are Laplacian-regularized. Also, hp_lambda is no scaled by len_reg so the penalty is the mean of absolute differences.
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
