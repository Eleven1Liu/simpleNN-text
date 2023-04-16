# README

## TBD

### Debug
- Test Accuracy
    - Simple-NN Text: 0.07
    - LibMultiLabel: 0.81

### Data
(He-Zhe)
- [ ] cannot read 'y' in ledgar_colwise.mat and ledgar.mat (OK in *.t.mat)
- [ ] there are some missing labels in sampled data so that matlab can't pass the data check
- [ ] Z format:  假設 N = max length, d = embed dim,
    ```
    Z =
    [ z11, z12, .. z1N,
    z21, z22, ... z2N,
    ...
    zd1, zd2, ..., zdN ]
    ```
    dump 出來 row vector 要是 row-wise:
    ```
    [z11, ..., z1N, z21, ... , z2N, ... zdN]
    ```

### Training
- [x] CNN: 2D to 1D
- [x] P4-P8: phiZ index (find_index_phiZ)
- [x] h*h -> 1*h
- [ ] P7: ignore zero padding (?) padding with LibMultiLabel now
- [ ] Train with Adam
- [ ] Clean code, remove unused args (a_in?)
- [ ] (low-priority) Concat different filter szs