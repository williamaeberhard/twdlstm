twdlstm: PyTorch code for training an LSTM NN on TWD_norm series
----------------------------------------------------------------

### TODO

* [x] loop over series list in config yaml, create one array for all data
* [ ] add in config whether to use CPU or GPU


### Version history

This is twdlstm version 0.3. Change log:
* v0.3: completely changed tr and va subsets. Entire time window (set by nT length in config) is split in batches of size batch_len (supplied in config), tr and va batches are randomly selected over all series with va size being specified by prop_va in config, and tr and va losses are only being evaluated at last loss_hor observations (also supplied in config, should stay 1 though otherwise overlap) within each batch.
* v0.2.1:
  - fixed total number of parameters printed in log
  - added loss to config, choice betwen 'MSE' and 'MAE'.
* v0.2: arbitrary number of series can be supplied (>=2), loop over seriesvec
* v0.1.1: fixed datetime stamp for Europe/Zurich tz
* v0.1: initial release
