from evaluator import Evaluator
from experiment import Experiment
from model_space import ModelSpace

import nni.retiarii.strategy as strategy


if __name__=="__main__":
    model_space = ModelSpace()
    evaluator = Evaluator(num_epochs=3)
    search_strategy = strategy.RegularizedEvolution()
    experiment = Experiment(model_space, evaluator, search_strategy)
    experiment.run()
    experiment.export_top()
