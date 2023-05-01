# README

## Getting started
### Run simpleNN with the profiler
```matlab
profile on
example('-bsize 5 -s 2 -lr 0.1 -decay 0 -C inf -epoch_max 5 -momentum 0.9');
profile viewer
```

## Data (LEDGAR)
- Please download the full data from **linux4: /tmp2/d11922012/optdl2023/data** and put them in **MATLAB/data**.
- For toy examples, you can see the files after git clone.

|                     | Full                   |  Toy Examples     |
| ------------------- | ---------------------- | ----------------- |
|  initial weights    | ledgar_init.mat        |  -                |
|  train              | ledgar.mat             |  ledgar_toy.mat   |
|  test               | ledgar.t.mat           |  ledgar_toy.t.mat |