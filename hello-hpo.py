"""
HPO Quickstart with PyTorch
===========================
This tutorial optimizes the model in `official PyTorch quickstart`_ with auto-tuning.
The tutorial consists of 4 steps: 
1. Modify the model for auto-tuning.
2. Define hyperparameters' search space.
3. Configure the experiment.
4. Run the experiment.
.. _official PyTorch quickstart: https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
"""

search_space = {
    'features': {'_type': 'choice', '_value': [128, 256, 512, 1024]},
    'lr': {'_type': 'loguniform', '_value': [0.0001, 0.1]},
    'momentum': {'_type': 'uniform', '_value': [0, 1]},
}

from nni.experiment import Experiment
experiment = Experiment('local')
experiment.config.trial_command = '/usr/local/bin/python3 model-hpo.py'
experiment.config.trial_code_directory = '.'
experiment.config.search_space = search_space
experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
experiment.config.max_trial_number = 10
experiment.config.trial_concurrency = 2
experiment.run(8085)

input('Press enter to quit')
experiment.stop()
