from early_stop_evaluator import EarlyTerminationEvaluator
from evaluator import Evaluator
from experiment import Experiment

import os
import nni.retiarii.strategy as strategy
from mobilenetv3_mod import MobileNetV3Space
import argparse

parser = argparse.ArgumentParser(description='Neural Architecture Search for Low Inference Time CNNs')
parser.add_argument('--no_gpu', action='store_true',
                    help='Use CPU instead of GPU')
parser.add_argument('--concurrent_trials', default=4, type=int,
                    help='How many trials to run concurrently')
parser.add_argument('--search_epochs', default=3, type=int,
                    help='How many epochs to run on search')
parser.add_argument('--cull_ratio', default=.3, type=float,
                    help='What percentage of models to terminate early')
parser.add_argument('--max_population', default=30, type=int,
                    help='Population size for the Regularized Evolutionary search')
parser.add_argument('--sample_size', default=7, type=int,
                    help='Sample to use in the Regularized Evolutionary search')
parser.add_argument('--n_trials', default=1000, type=int,
                    help='How many trials to run in total')
args = parser.parse_args()


if __name__=="__main__":
    model_space = MobileNetV3Space()
    if args.cull_ratio == 0:
        evaluator = Evaluator(num_epochs=args.search_epochs)
    else:
        evaluator = EarlyTerminationEvaluator(num_epochs=args.search_epochs, cull_ratio=args.cull_ratio, max_population=args.max_population)
    search_strategy = strategy.RegularizedEvolution(population_size=args.max_population, cycles=args.n_trials-args.max_population, sample_size=args.sample_size)
    experiment = Experiment(model_space, evaluator, search_strategy, args.no_gpu, args.concurrent_trials, args.n_trials)
    try:
        os.remove("pastepacc.txt")
    except FileNotFoundError:
        pass
    experiment.run()
    experiment.export_top()
