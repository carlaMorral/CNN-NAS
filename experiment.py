from nni.retiarii.evaluator import FunctionalEvaluator
from nni.retiarii.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig


class Experiment:

    def __init__(self, model_space, evaluator, search_strategy):
        self.evaluator = FunctionalEvaluator(evaluator.evaluate_model)
        self.exp = RetiariiExperiment(model_space, self.evaluator, [], search_strategy)
        self._config()

    def _config(self):
        self.exp_config = RetiariiExeConfig('local')
        self.exp_config.experiment_name = 'mnist_search'
        self.exp_config.max_trial_number = 96
        self.exp_config.trial_concurrency = 1
        self.exp_config.trial_gpu_number = 4
        self.exp_config.training_service.use_active_gpu = True

    def run(self):
        self.exp.run(self.exp_config, 8082)

    def export_top(self):
        for model_dict in self.exp.export_top_models(formatter='dict'):
            print(model_dict)
