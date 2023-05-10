# CNN-NAS

## Description of the project

We leverage the NNI library (link: https://github.com/microsoft/nni) to run Neural Architecture Search in order to find a CNN that reduces inference time on GPUs while maintaining a good accuracy. We use Regularized Evolution to perform NAS (paper: https://arxiv.org/pdf/1802.01548.pdf) on the MobileNetV3 (paper: https://arxiv.org/pdf/1905.02244.pdf) search space. We also try a few optimizations (reduced number of epochs and early termination evaluation strategies) to reduce the search time so it becomes possible to perform NAS on less powerful clusters.

## Outline of the code repository

## How to run
Install the requirements:
`pip3 install .`
`python3 setup.py install`

Run the project:
`python3 main.py`
You can add the following flags to the above command to modify the search:
`--no_gpu` runs the models on CPU in case you run it in a cluster with no GPUs
`--concurrent_trials t` will run `t` trials concurrently. (default: 4) Very useful when the cluster contains multiple GPUs.
`--search epochs e` will train each model for `e` epochs during search. (default: 3)
`--cull_ratio c` will use the early termination evaluator with the appropriate `c` if `c != 0` (default: 0.3)
`--max_population p` sets the maximum population for the Regularized Evolution search. (default: 30)
`--sample_size s` sets the sample size for the Regularized Evolution search. (default: 7)
`--n_trials t` will run `t` trials total. (default: 1000)

Note that, due to a bug in NNI, the program will hang after the search is finished. Instead, you can use `^C` to end the program at any time and use the following script:
`python3 read_trial_data.py <experiment ID>`
to view a summary of the results of the experiment. For this program to work correctly, the folders `CNN-NAS` and `nni-experiments` need to be on the same level.

You can also evaluate a model by training it for 50 epochs using the following script:
`python3 test_model.py <path to trial>`
where `<path to trial>` should be inside the folder `nni-experiments` and point to the folder corresponding to the trial you want to test.

## Experimental results
