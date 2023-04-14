All results maybe reproduced by executing [reproduce_results.py](https://github.com/IPMI-ICNS-UKE/sparsam/blob/master/reproducability/reproduce_results.py).
```bash
python sparsam/reproducability/reproduce_results.py /path/to/images/ data_set_name --options
```

data_set_name can be either one of endo, bm, isic.

Options include:
- **--trainings_mode**:  Which results to reproduce. The mode is provided as composed string. If "ssl" is part of the string. The model is going to be pretrained with DINO otherwise precomputed weights are used. Valid useful options are: "ssl,classifier", "ssl,linear", "classifier", "linear", and "supervised". In principle all combinations are valid and will be executed.
- **--min_class_size**: only consider classes, where the training_samples exceed the provided min class_size. Default is None.
- **--eval_class_sizes**: Which class sizes to evaluate. Multiple are possible. e.g. "-e 10 -e 250. Evaluates the test set for 10 and 250 samples per class. Default is "1, 5, 10, 25, 50, 100, 250, 500, 1000"
- **--n_iterations**: How often to repeat experiments (does not include self supervised training).
- **--device**: Which GPU to train on
- **--batch_size**: batch size on gpu
- **-- num_workers**: How many workers to use for data loading
- **-- gradient_checkpointing**: Whether to use gradient checkpointing. GC trades memory for computation time, which leads to a smaller memory footprint with a slower training. Resulting in a larger possible batch size
- **--load_machine_param_from_yaml**: Path to the corresponding yaml. If "True" the default path is used.

All options can also be listed with:
```bash
python sparsam/reproducability/reproduce_results.py --help
```
