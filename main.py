from early_stop_evaluator import Evaluator
from experiment import Experiment

import os
import nni.retiarii.strategy as strategy
from mobilenetv3_mod import MobileNetV3Space


if __name__=="__main__":
    model_space = MobileNetV3Space()
    evaluator = Evaluator(num_epochs=3, cull_ratio=.4, max_population=30)
    search_strategy = strategy.RegularizedEvolution(population_size=30, cycles=970, sample_size=7)
    experiment = Experiment(model_space, evaluator, search_strategy)
    try:
        os.remove("pastepacc.txt")
    except FileNotFoundError:
        pass
    experiment.run()
    experiment.export_top()
