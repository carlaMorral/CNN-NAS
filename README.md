# CNN-NAS

## Description of the project

We leverage the NNI library (link: https://github.com/microsoft/nni) to run Neural Architecture Search in order to find a CNN that reduces inference time on GPUs while maintaining a good accuracy. We use Regularized Evolution to perform NAS (paper: https://arxiv.org/pdf/1802.01548.pdf) on the MobileNetV3 (paper: https://arxiv.org/pdf/1905.02244.pdf) search space. We also try a few optimizations (reduced number of epochs and early termination evaluation strategies) to reduce the search time so it becomes possible to perform NAS on less powerful clusters.

## Outline of the code repository

`main.py` contains the code necessary to run the search algorithm.

`experiment.py` contains the code necessary to define an experiment.

`evaluator.py` contains the code for the evaluator without early termination, while `early_stop_evaluator.py` contains the code for the evaluator with early termination.

`mobilenetv3_mod.py` contains the modification to the NNI library to make our experiment work.

`read_trial_data.py` contains the script to obtain a summary of an experiment after finishing it.

`setup.py` installs the necessary libraries.

`test_evaluator.py` contains the evaluator that's used in `test_model.py`.

`test_model.py` trains one model from an experiment for a larger number of epochs and gives more detailed statistics.

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

We tested some combinations of epochs (1, 3, 5, 10) and cull rates (0%, 15%, 30%, 45%, 66%). All experiments ran for 250 trials (except those with 10 epochs, see below), the search time was recorded, and the model with the best metric in each experiment was trained for 50 epochs to obtain accuracy and inference time. Dashed cells indicate experiments that weren't performed. In particular, the experiments with 1 epoch and cull rate greater than 0% are redundant as the algorithm cannot terminate any sooner than 1 epoch.

The results of the experiments of cull rate = 0.66 aren't shown because, with the cull rate so high, the evolution algorithm wasn't able to improve the models from the initial population so it didn't make sense to spend computational resources to test a model that we already knew was going to be very bad compared to the others. The experiments with 10 epochs only ran for about half the trials as all the other experiments because they were taking far too long, so the results aren't really comparable to the other experiments but we include them here for completeness. The reported search time is the amount of time it would've taken to perform 250 trials at the rate that the rest of trials were performed.

The search time refers to the total amount of the time that went into the NAS.The inference speedups are calculated with respect to a baseline model obtained by running NAS for 250 trials with 3 epochs per model and a cull rate of 30%. This baseline model also had an 83.73% accuracy after being trained for 50 epochs. The search for the baseline model took 3h 06m.

|            |                   | 0% cull          | 15% cull         | 30% cull         | 45% cull         |
|------------|-------------------|------------------|------------------|------------------|------------------|
| 1 epoch    | Accuracy          | 81.45            | -                | -                | -                |
|            | Search time       | 1h 24m           | -                | -                | -                |
|            | Inference speedup | 1.06             | -                | -                | -                |
| 3 epochs   | Accuracy          | 84.27            | 82.95            | 84.03            | 83.87            |
|            | Search time       | 2h 59m           | 2h 31m           | 2h 39m           | 2h 49m           |
|            | Inference speedup | 1.01             | 0.81             | 1.15             | 1.08             |
| 5 epochs   | Accuracy          | 83.34            | -                | 83.16            | 83.78            |
|            | Search time       | 4h 10m           | -                | 4h 03m           | 5h 18m           |
|            | Inference speedup | 1.22             | -                | 0.96             | 0.84             |
| 10 epochs* | Accuracy          | 82.53            | -                | 80.92            | -                |
|            | Search time       | 9h 26m           | -                | 10h 11m          | -                |
|            | Inference speedup | 0.98             | -                | 0.95             | -                |

One big limitation of these experiments is that they're sensitive to how good the initial population is, as it's selected at random in each experiment. Two common ways to control for this are to run a larger number of trials or use a set seed. The former wasn't possible due to computational resource constraints and the latter wasn't possible because NNI doesn't support set seeds. (to the best of our knowledge. We spent hours looking.) This means that there will be a fair amount of varaiability between experiments.

As discussed previously, a cull rate of 66% is too high to make the evolutionary search work, but in the 3 epochs case we get good results with a cull rate of 30% and decent results with a cull rate of 45%, indicating that the bottom 30-45% of models don't affect much the evolutionary search and we can safely terminate them early to decrease the search time. However, the decrease in search time we get isn't too big. In the 5 epochs case we get worse inference speedups with a higher cull rate, but that could be attributed to a worse starting population.

The biggest factor that affects search time by far is the number of epochs during search. With only 1 epoch the model we get is a bit weak in terms of accuracy, probably because training on 1 epoch isn't a great indicator of how a model will perform after training on 50 epochs, but judging by the results, 3 epochs seems sufficient. This allows for a big reduction in the search time compared to the 25 epochs per model they use in the MobileNetV3 paper.

Another factor that affects the search time is the inference speedup. This makes sense, as the best models get replicated and mutated more often, and if the best model is also fast this means that the training will take less time, which decreases the overall search time as training is the most computationally expensive part of the process. If we look at the results, there is a negative correlation between the inference speedup and the search time, but it's not perfect because  it's possible that the best model was found really late into the search and for most of the search, some other model with very different speedup was the best one and that was the one that got replicated and mutated the most.

In conclusion, we can deduce that it should be okay to reduce the number of epochs during NAS to about 3 and still get a metric that's representative of what happens after 50 epochs, and that early termination is a sound idea but it only reduces search time a little. However, these results have a lot of variability to them that can be explained by different starting populations and further work should be done in running more experiments to confirm our findings. Furthermore, since models with faster inference times also make the search take less time, it might be worth it to try to include inference time in the metric even if we only care about accuracy, as making the search faster means we can try more models and have a better chance of finding one of the best ones.
