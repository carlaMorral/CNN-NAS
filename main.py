from evaluator import Evaluator
from experiment import Experiment
from model_space import ModelSpace

import os
import nni.retiarii.strategy as strategy
from nni.retiarii.hub.pytorch import MobileNetV3Space


if __name__=="__main__":
    model_space = MobileNetV3Space()
    evaluator = Evaluator(num_epochs=3)
    search_strategy = strategy.RegularizedEvolution()
    experiment = Experiment(model_space, evaluator, search_strategy)
    experiment.run()
    experiment.export_top()
    try:
        os.remove("avgep1acc.txt")
    except FileNotFoundError:
        pass
