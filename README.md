# README


## Getting started
### Run simpleNN with the profiler
```matlab
profile on
example('-bsize 5 -s 2 -lr 0.1 -decay 0 -C inf -epoch_max 5 -momentum 0.9');
profile viewer
```

## Data
### Train / Test data
N = max length, d = embed dim
|                     |  Description         |
| ------------------- | -------------------- |
|  dataset            | LEDGAR               |
|  shape              | (100, 3000) = (# of instances, $d \times N) $    |
|  file path          | **train**: data/ledger_toy.mat , **test**: data/ledger_toy.t.mat  |
|  row-wise $Z^{in,i}$ |  $[z_{11}, ..., z_{1N}, z_{21}, ... , z_{2N}, ... z_{dN}]$ |

### Init weights from LibMultiLabel
- data/ledgar_init_toy.mat

## Code Changes
### SimpleNN Config
- config/ledgar_toy.config

### Forward
- **cnn/find_index_phiZ.m**
    - $h \times h$ to $1 \times h$
    - set `a_out` to 1
- **cnn/maxpooling.m**, **cnn/padding_and_phiZ.m**
    - $h \times h$ to $1 \times h$
- **cnn/train.m**
    - load init weight for convolutional layer and linear layer (provide students the code)
    - update shape of convolutional layer (`ht_* = 1`)

### Backward
- **cnn/lossgrad_subset.m**: to be determined
- **opt/adam.m**, **opt/sgd.m**: no shuffle

### Others
- [ ] Clean code, remove unused args (`a_in`?)

### LibMultiLabel config
```yaml=
max_seq_length: 10
shuffle: false
val_size: 0

model_name: KimCNN
network_config:
  embed_dropout: 0.2
  encoder_dropout: 0 # 0.2
  filter_sizes: [2] # [2, 4, 8]
  num_filter_per_size: 128 # filter channels
```