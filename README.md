# README

## Getting started
### Run with the profiler
```matlab
profile on
example("-bsize 100 -s 2 -lr 0.1 -decay 0 -C inf -epoch_max 10", 0, "config/ledgar_toy.config");
profile viewer
```

## Data (LEDGAR)

### SimpleNN
- Please download the full data from **linux4: /tmp2/d11922012/optdl2023/data** and put them in **MATLAB/data**.
- For toy examples, you can see the files after git clone.

|                     | Full                   |  Toy Examples         |
| ------------------- | ---------------------- | --------------------- |
|  initial weights    | ledgar_init.mat (optional)        |  ledgar_init_toy.mat  |
|  train              | ledgar.mat             |  ledgar_toy.mat       |
|  test               | ledgar.t.mat           |  ledgar_toy.t.mat     |

### LibMultiLabel
- The config file and the toy examples for running [LibMultiLabel](https://github.com/ASUS-AICS/LibMultiLabel/tree/profiler) is in **data/LibMultiLabel**.
